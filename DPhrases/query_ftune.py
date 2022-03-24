import json
import argparse
import torch
import os
import random
import numpy as np
import requests
import logging
import math
import copy
import string
import faiss

from time import time
from tqdm import tqdm
from densephrases.utils.squad_utils import get_question_dataloader
from densephrases.utils.single_utils import load_encoder
from densephrases.utils.open_utils import load_phrase_index, get_query2vec, load_qa_pairs
from densephrases.utils.eval_utils import drqa_exact_match_score, drqa_regex_match_score, \
    drqa_metric_max_over_ground_truths
from eval_phrase_retrieval import evaluate
from densephrases import Options

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train_query_encoder(args, mips=None):
    # Freeze one for MIPS
    device = 'cuda' if args.cuda else 'cpu'
    logger.info("Loading pretrained encoder: this one is for MIPS (fixed)")
    pretrained_encoder, tokenizer, _ = load_encoder(device, args)

    # Train a copy of it
    logger.info("Copying target encoder")
    target_encoder = copy.deepcopy(pretrained_encoder)

    # MIPS
    if mips is None:
        mips = load_phrase_index(args)

    # Optimizer setting
    def is_train_param(name):
        if name.endswith(".embeddings.word_embeddings.weight"):
            logger.info(f'freezing {name}')
            return False
        return True

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [
            p for n, p in target_encoder.named_parameters() \
            if not any(nd in n for nd in no_decay) and is_train_param(n)
        ],
        "weight_decay": 0.01,
    }, {
        "params": [
            p for n, p in target_encoder.named_parameters() \
            if any(nd in n for nd in no_decay) and is_train_param(n)
        ],
        "weight_decay": 0.0
    },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    step_per_epoch = math.ceil(len(load_qa_pairs(args.train_path, args)[1]) / args.per_gpu_train_batch_size)
    # Get the total steps and prewarmup steps
    t_prewarm = int(step_per_epoch // args.gradient_accumulation_steps * args.num_firsthop_epochs)
    t_total = int(step_per_epoch // args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info(f"Train for {t_total} iterations")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    eval_steps = math.ceil(len(load_qa_pairs(args.dev_path, args)[1]) / args.eval_batch_size)
    logger.info(f"Test takes {eval_steps} iterations")

    # Train arguments
    args.per_gpu_train_batch_size = int(args.per_gpu_train_batch_size / args.gradient_accumulation_steps)
    best_acc = -1000.0
    cnt = 0 # Counts uptill epochs where the training strategy needs a change 

    if cnt < args.num_firsthop_epochs:
        # Warm-up the model with a first-hop retrieval objective
        for ep_idx in range(int(args.num_firsthop_epochs)):
            # Training
            total_loss = 0.0
            total_accs = []
            total_accs_k = []

            # Load training dataset
            q_ids, questions, answers, titles, full_answers = load_qa_pairs(args.train_path, args, shuffle=True)
            pbar = tqdm(get_top_phrases(
                mips, q_ids, questions, answers, titles, pretrained_encoder, tokenizer,
                args.per_gpu_train_batch_size, args, final_answers)
            )

            for step_idx, (q_ids, questions, answers, titles, outs, final_answers, outs_single) in enumerate(pbar):
                train_dataloader, _, _ = get_question_dataloader(
                    questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
                )
                svs, evs, tgts, p_tgts = annotate_phrase_vecs(mips, q_ids, questions, answers, titles, outs, args)

                target_encoder.train()
                svs_t = torch.Tensor(svs).to(device)
                evs_t = torch.Tensor(evs).to(device)
                tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts]
                p_tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in p_tgts]

                # Train query encoder
                assert len(train_dataloader) == 1
                for batch in train_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    loss, accs = target_encoder.train_query(
                        input_ids_=batch[0], attention_mask_=batch[1], token_type_ids_=batch[2],
                        start_vecs=svs_t,
                        end_vecs=evs_t,
                        targets=tgts_t,
                        p_targets=p_tgts_t,
                    )

                    # Optimize, get acc and report
                    if loss is not None:
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        total_loss += loss.mean().item()
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), args.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        target_encoder.zero_grad()

                        pbar.set_description(
                            f"Ep {ep_idx + 1} Tr loss: {loss.mean().item():.2f}, acc: {sum(accs) / len(accs):.3f}"
                        )

                    if accs is not None:
                        total_accs += accs
                        total_accs_k += [len(tgt) > 0 for tgt in tgts_t]
                    else:
                        total_accs += [0.0] * len(tgts_t)
                        total_accs_k += [0.0] * len(tgts_t)

            step_idx += 1
            logger.info(
                f"Avg train loss ({step_idx} iterations): {total_loss / step_idx:.2f} | train " +
                f"acc@1: {sum(total_accs) / len(total_accs):.3f} | acc@{args.top_k}: {sum(total_accs_k) / len(total_accs_k):.3f}"
            )

            # Evaluation
            new_args = copy.deepcopy(args)
            new_args.top_k = 10
            new_args.save_pred = False
            new_args.test_path = args.dev_path
            dev_em, dev_f1, dev_emk, dev_f1k = evaluate(new_args, mips, target_encoder, tokenizer)
            logger.info(f"Develoment set acc@1: {dev_em:.3f}, f1@1: {dev_f1:.3f}")

            # Save best model
            if dev_em > best_acc:
                best_acc = dev_em
                save_path = args.output_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                target_encoder.save_pretrained(save_path)
                logger.info(f"Saved best model with acc {best_acc:.3f} into {save_path}")

            if (ep_idx + 1) % 1 == 0:
                logger.info('Updating pretrained encoder')
                pretrained_encoder = copy.deepcopy(target_encoder)

        print()
        logger.info(f"Best model has acc {best_acc:.3f} saved as {save_path}")
    else:
        # Run the full-training strategy accounting for loss from both the epochs 
        for ep_idx in range(int(args.num_firsthop_epochs),int(args.num_train_epochs)):
            total_loss = 0.0
            loss_hop1    = 0.0
            loss_hop2    = 0.0
            total_accs_1  = []
            total_accs2   = []
            total_accs_k1 = []
            total_accs_k2 = []
            # Get questions and corresponding answers with other metadata
            q_ids, questions, answers, titles, full_answers = load_qa_pairs(args.train_path, args, shuffle=True)

            # Perform first hop search 
            pbar_hop1 = tqdm(get_top_phrases(
                mips, q_ids, questions, answers, titles, pretrained_encoder, tokenizer,
                args.per_gpu_train_batch_size, args, final_answers)
            )
            for hop_step_idx, (q_ids, questions, answers, titles, outs, final_answers, outs_single) in enumerate(pbar_hop1):

                train_dataloader, _, _ = get_question_dataloader(
                    questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
                )

                # Get first hop level start & end vectors alongwith targets
                svs, evs, tgts, p_tgts = annotate_phrase_vecs(mips, q_ids, questions, answers, titles, outs, args)
                # Create updated query for second hop search
                upd_questions = []
                for (query, out_single) in zip(questions, outs_single):
                    upd_questions.append(query + " "+ out_single)
                # Use these updated query to perform another round of search 
                # Pass final answers in place of answers
                # To do: Think and Add condition related to aggregation so that phrase level retrieval occurs
                # Might need to modify get_top_phrases function and add a conditional here 
                pbar_hop2 = tqdm(get_top_phrases(
                mips, q_ids, upd_questions, final_answers, titles, pretrained_encoder, tokenizer,
                args.per_gpu_train_batch_size, args, final_answers)

                # Get second hop level start & end vectors alongwith final targets 
                # Passing questions/upd_questions should not have an impact (check question)
                svs_sec, evs_sec, tgts_sec, p_tgts_sec = annotate_phrase_vecs(mips, q_ids, upd_questions, final_answers, titles, outs, args)

                target_encoder.train()

                svs_t = torch.Tensor(svs).to(device)
                evs_t = torch.Tensor(evs).to(device)
                tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts]
                p_tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in p_tgts]

                svs_t_sec = torch.Tensor(svs_sec).to(device)
                evs_t_sec = torch.Tensor(evs_sec).to(device)
                tgts_t_sec = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts_sec]
                p_tgts_t_sec = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in p_tgts_sec]



                # Batch level computations for loss functions 
                assert len(train_dataloader) == 1
                for batch in train_dataloader:
                    loss_hop1, accs_1 = target_encoder.train_query(
                        input_ids_=batch[0], attention_mask_=batch[1], token_type_ids_=batch[2],
                        start_vecs=svs_t,
                        end_vecs=evs_t,
                        targets=tgts_t,
                        p_targets=p_tgts_t,
                    )

                    loss_hop1, accs_2 = target_encoder.train_query(
                        input_ids_=batch[0], attention_mask_=batch[1], token_type_ids_=batch[2],
                        start_vecs=svs_t_sec,
                        end_vecs=evs_t_sec,
                        targets=tgts_t_sec,
                        p_targets=p_tgts_t_sec,
                    )

                    if loss_hop1 and loss_hop2 is not None:  
                        if args.gradient_accumulation_steps > 1:
                            loss =  (args.weight1* loss_hop1 + (args.weight2)*loss_hop2)/args.gradient_accumulation_steps #Add weight param in args (Todo)
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        total_loss += loss.mean().item()
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), args.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule if reqired (ToDo)
                        target_encoder.zero_grad()

                        pbar.set_description(
                            f"Ep {ep_idx + 1} Tr loss: {loss.mean().item():.2f}, acc1: {sum(accs1) / len(accs1):.3f}, acc2: {sum(accs2) / len(accs2):.3f}"
                        )
                    
                    if accs1 is not None:
                        total_accs1 += accs1
                        total_accs_k += [len(tgt) > 0 for tgt in tgts_t]
                    else:
                        total_accs_1 += [0.0] * len(tgts_t)
                        total_accs_k += [0.0] * len(tgts_t)
                    
                    if accs2 is not None:
                        total_accs2 += accs2
                        total_accs_k2 += [len(tgt) > 0 for tgt in tgts_t_sec]
                    else:
                        total_accs_2 += [0.0] * len(tgts_t_sec)
                        total_accs_k2 += [0.0] * len(tgts_t_sec)
                
            hop_step_idx += 1
            logger.info(
                f"Avg train loss ({hop_step_idx} iterations): {total_loss / step_idx:.2f} | train " +
                f"acc@1: {sum(total_accs2) / len(total_accs2):.3f} | acc@{args.top_k}: {sum(total_accs_k2) / len(total_accs_k2):.3f}"
            )

            # Evaluation
            new_args = copy.deepcopy(args)
            new_args.top_k = 10
            new_args.save_pred = False
            new_args.test_path = args.dev_path
            dev_em, dev_f1, dev_emk, dev_f1k = evaluate(new_args, mips, target_encoder, tokenizer)
            logger.info(f"Develoment set acc@1: {dev_em:.3f}, f1@1: {dev_f1:.3f}")

            # Save best model
            if dev_em > best_acc:
                best_acc = dev_em
                save_path = args.output_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                target_encoder.save_pretrained(save_path)
                logger.info(f"Saved best model with acc {best_acc:.3f} into {save_path}")

            if (ep_idx + 1) % 1 == 0:
                logger.info('Updating pretrained encoder')
                pretrained_encoder = copy.deepcopy(target_encoder)

        print()
        logger.info(f"Best model has acc {best_acc:.3f} saved as {save_path}")

    # TODO: Add a "full" query-tuning training loop, i.e. joint first-hop + final-answer training
    # Update: Added a boilerplate code for iteration n fine-tuning development script
    # TO DO: Run & Reverify the pipeline and resolve known & unknown issues 
    # Known Issues: Initially assumed outs is a list of top_k phrases but takes dictionary form (see annotate phrase vecs dummy)
    # Change in args and see what internal parameters to change with args so that second hop runs and see if need to redefine scheduler (mostly not)
    

def get_top_phrases(mips, q_ids, questions, answers, titles, query_encoder, tokenizer, batch_size, args, final_answers):
    # Search
    step = batch_size
    phrase_idxs = []
    search_fn = mips.search
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )
    for q_idx in tqdm(range(0, len(questions), step)):
        outs = query2vec(questions[q_idx:q_idx + step])
        start = np.concatenate([out[0] for out in outs], 0)
        end = np.concatenate([out[1] for out in outs], 0)
        query_vec = np.concatenate([start, end], 1)

        outs = search_fn(
            query_vec,
            q_texts=questions[q_idx:q_idx + step], nprobe=args.nprobe,
            top_k=args.top_k, return_idxs=True,
            max_answer_length=args.max_answer_length, aggregate=args.aggregate, agg_strat=args.agg_strat,
        )

        # Get single top results
        outs_single = []
        for out in outs:
            outs_single.append(out[0])

        yield (
            q_ids[q_idx:q_idx + step], questions[q_idx:q_idx + step], answers[q_idx:q_idx + step],
            titles[q_idx:q_idx + step], outs, final_answers[q_idx:q_idx + step], outs_single
        )


def annotate_phrase_vecs(mips, q_ids, questions, answers, titles, phrase_groups, args):
    assert mips is not None
    batch_size = len(answers)

    # Phrase groups are in size of [batch, top_k, values]
    # phrase_groups = [[(
    #     out_['doc_idx'], out_['start_idx'], out_['end_idx'], out_['answer'],
    #     out_['start_vec'], out_['end_vec'], out_['context'], out_['title'])
    #     for out_ in out] for out in outs
    # ]
    dummy_group = {
        'doc_idx': -1,
        'start_idx': 0, 'end_idx': 0,
        'answer': '',
        'start_vec': np.zeros(768),
        'end_vec': np.zeros(768),
        'context': '', 'title': ['']
    }

    # Pad phrase groups (two separate top-k coming from start/end, so pad with top_k*2)
    for b_idx, phrase_idx in enumerate(phrase_groups):
        while len(phrase_groups[b_idx]) < args.top_k * 2:
            phrase_groups[b_idx].append(dummy_group)
        assert len(phrase_groups[b_idx]) == args.top_k * 2

    # Flatten phrase groups
    flat_phrase_groups = [phrase for phrase_group in phrase_groups for phrase in phrase_group]
    doc_idxs = [int(phrase_group['doc_idx']) for phrase_group in flat_phrase_groups]
    start_vecs = [phrase_group['start_vec'] for phrase_group in flat_phrase_groups]
    end_vecs = [phrase_group['end_vec'] for phrase_group in flat_phrase_groups]

    # stack vectors
    start_vecs = np.stack(start_vecs)
    end_vecs = np.stack(end_vecs)
    zero_mask = np.array([[1] if doc_idx >= 0 else [0] for doc_idx in doc_idxs])
    start_vecs = start_vecs * zero_mask
    end_vecs = end_vecs * zero_mask

    # Reshape
    start_vecs = np.reshape(start_vecs, (batch_size, args.top_k * 2, -1))
    end_vecs = np.reshape(end_vecs, (batch_size, args.top_k * 2, -1))

    # Dummy targets
    targets = [[None for phrase in phrase_group] for phrase_group in phrase_groups]
    p_targets = [[None for phrase in phrase_group] for phrase_group in phrase_groups]

    # TODO: implement dynamic label_strategy based on the task name (label_strat = dynamic)

    # Annotate for L_phrase
    if 'phrase' in args.label_strat.split(','):
        match_fns = [
            drqa_regex_match_score if args.regex or ('trec' in q_id.lower()) else drqa_exact_match_score for q_id in
            q_ids
        ]
        targets = [
            [drqa_metric_max_over_ground_truths(match_fn, phrase['answer'], answer_set) for phrase in phrase_group]
            for phrase_group, answer_set, match_fn in zip(phrase_groups, answers, match_fns)
        ]
        targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]

    # Annotate for L_doc
    if 'doc' in args.label_strat.split(','):
        p_targets = [
            [any(phrase['title'][0].lower() == tit.lower() for tit in title) for phrase in phrase_group]
            for phrase_group, title in zip(phrase_groups, titles)
        ]
        p_targets = [[ii if val else None for ii, val in enumerate(target)] for target in p_targets]

    return start_vecs, end_vecs, targets, p_targets


if __name__ == '__main__':
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_index_options()
    options.add_retrieval_options()
    options.add_data_options()
    options.add_qsft_options()
    args = options.parse()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_mode == 'train_query':
        # Train
        mips = load_phrase_index(args)
        train_query_encoder(args, mips)

        # Eval
        args.load_dir = args.output_dir
        logger.info(f"Evaluating {args.load_dir}")
        args.top_k = 10
        evaluate(args, mips)

    else:
        raise NotImplementedError
