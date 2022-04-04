import torch
import os
import random
import numpy as np
import logging
import math
import copy
from tqdm import tqdm
from IPython import embed

from densephrases.utils.squad_utils import get_question_dataloader
from densephrases.utils.single_utils import load_encoder
from densephrases.utils.open_utils import load_phrase_index, get_query2vec, load_qa_pairs
from densephrases.utils.eval_utils import drqa_exact_match_score, drqa_regex_match_score, \
    drqa_metric_max_over_ground_truths, drqa_substr_match_score, drqa_substr_f1_match_score
from eval_phrase_retrieval import evaluate
from densephrases import Options

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

run_id = str(__import__('calendar').timegm(__import__('time').gmtime()))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_run():
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_index_options()
    options.add_retrieval_options()
    options.add_data_options()
    options.add_qsft_options()
    args = options.parse()

    # Create output dir
    if args.output_dir is None:
        raise ValueError("Missing argument: --output_dir")
    save_path = os.path.join(args.output_dir, run_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Setup logging
    root_logger = logging.getLogger()
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')
    fh = logging.FileHandler(f"{save_path}/log.txt", mode="a", delay=False)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    print("\n\n")
    logger.info(f"Starting with RUN_ID={run_id}")
    print("\n\n")

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, save_path

def is_train_param(name):
    if name.endswith(".embeddings.word_embeddings.weight"):
        logger.info(f'freezing {name}')
        return False
    return True


def shuffle_data(data):
    q_ids, questions, answers, titles, final_answers, final_titles = data
    qa_pairs = list(zip(q_ids, questions, answers, titles, final_answers, final_titles))
    random.shuffle(qa_pairs)
    q_ids, questions, answers, titles, final_answers, final_titles = zip(*qa_pairs)
    logger.info(f'Shuffling QA pairs')
    return q_ids, questions, answers, titles, final_answers, final_titles


def get_optimizer_scheduler(encoder, args, train_len, dev_len, n_epochs):
    # Optimizer settings
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [
            p for n, p in encoder.named_parameters() \
            if not any(nd in n for nd in no_decay) and is_train_param(n)
        ],
        "weight_decay": 0.01,  # Training parameters that should be decayed
    }, {
        "params": [
            p for n, p in encoder.named_parameters() \
            if any(nd in n for nd in no_decay) and is_train_param(n)
        ],
        "weight_decay": 0.0  # Training parameters that should not be decayed
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # Scheduler settings
    step_per_epoch = math.ceil(train_len / args.per_gpu_train_batch_size)
    # Get total training steps
    t_total = int(step_per_epoch // args.gradient_accumulation_steps * n_epochs)
    logger.info(f"Training for {t_total} iterations")
    scheduler = get_linear_schedule_with_warmup(  # "warmup" here refers to the lr warmup
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    eval_steps = math.ceil(dev_len / args.eval_batch_size)
    logger.info(f"DEV eval takes {eval_steps} iterations")

    return optimizer, scheduler


def train_query_encoder(args, save_path, mips=None, init_dev_acc=None):
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

    # Load train and dev data
    train_qa_pairs = load_qa_pairs(args.train_path, args, multihop=True)
    dev_qa_pairs = load_qa_pairs(args.dev_path, args, multihop=True)

    if not args.skip_warmup:
        logger.info(f"Starting warmup training")
        # Initialize optimizer & scheduler
        optimizer, scheduler = get_optimizer_scheduler(target_encoder, args, len(train_qa_pairs[1]),
                                                       len(dev_qa_pairs[1]),
                                                       args.num_firsthop_epochs)

        # Train arguments
        args.per_gpu_train_batch_size = int(args.per_gpu_train_batch_size / args.gradient_accumulation_steps)
        best_acc = -1000.0 if init_dev_acc is None else init_dev_acc
        best_epoch = -1

        # Warm-up the model with a first-hop retrieval objective
        for ep_idx in range(int(args.num_firsthop_epochs)):
            # Training
            total_loss = 0.0
            total_accs = []
            total_accs_k = []

            # Load training dataset
            q_ids, questions, answers, titles, final_answers, final_titles = shuffle_data(train_qa_pairs)

            # Progress bar
            pbar = tqdm(get_top_phrases(
                mips, q_ids, questions, answers, titles, pretrained_encoder, tokenizer,  # encoder updated every epoch
                args.per_gpu_train_batch_size, args, final_answers, final_titles, agg_strat=args.warmup_agg_strat)
            )

            for step_idx, (q_ids, questions, answers, titles, outs, final_answers, final_titles) in enumerate(pbar):
                # INFO: `outs` contains topk phrases for each query

                train_dataloader, _, _ = get_question_dataloader(
                    questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
                )
                svs, evs, tgts, p_tgts = annotate_phrase_vecs(mips, q_ids, questions, answers, titles, outs, args,
                                                              label_strat=args.warmup_label_strat)

                target_encoder.train()
                # Start vectors
                svs_t = torch.Tensor(svs).to(device)  # shape: (bs, 2*topk, hid_dim)
                # End vectors
                evs_t = torch.Tensor(evs).to(device)
                # Target indices for phrases
                tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts]
                # Target indices for doc
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
                    )  # INFO: `accs` is the accuracy of the top-1 probability prediction being a valid gold target

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
                f"acc@1: {sum(total_accs) / len(total_accs):.3f} | " +
                f"acc@{args.top_k}: {sum(total_accs_k) / len(total_accs_k):.3f}"
            )

            # Dev evaluation
            print("\n")
            logger.info("Evaluating on the DEV set")
            print("\n")
            new_args = copy.deepcopy(args)
            new_args.top_k = 10
            new_args.save_pred = False
            new_args.test_path = args.dev_path
            eval_res = evaluate(new_args, mips, target_encoder, tokenizer, firsthop=True)
            dev_em, dev_f1, dev_emk, dev_f1k = eval_res[0]
            evid_dev_em, evid_dev_f1, evid_dev_emk, evid_dev_f1k = eval_res[1]
            joint_substr_f1, joint_substr_f1k = eval_res[2]
            # Warm-up dev metric to use for picking the best epoch
            dev_metrics = {
                "phrase": dev_em,
                "evidence": evid_dev_f1,
                "joint": joint_substr_f1
            }
            logger.info(f"Dev set acc@1: {dev_em:.3f}, f1@1: {dev_f1:.3f}")
            logger.info(f"Dev set (evidence) acc@1: {evid_dev_em:.3f}, f1@1: {evid_dev_f1:.3f}")
            logger.info(f"Dev set (joint) acc@1: {joint_substr_f1:.3f}")

            # Save best model
            # TODO: Decide if evaluation should be on phrase or sentence (i.e., dev_em or evid_dev_em)
            if dev_metrics[args.warmup_dev_metric] > best_acc:
                best_acc = dev_metrics[args.warmup_dev_metric]
                best_epoch = ep_idx
                target_encoder.save_pretrained(save_path)
                logger.info(f"Saved best model with ({args.warmup_dev_metric}) acc. {best_acc:.3f} into {save_path}")

            # TODO: Eliminate pretrained_encoder and use only target_encoders as suggested by authors
            logger.info('Updating pretrained encoder')
            pretrained_encoder = copy.deepcopy(target_encoder)
        print()
        logger.info(f"Best model (epoch {best_epoch}) with accuracy {best_acc:.3f} saved at {save_path}")

    if not args.warmup_only:
        logger.info(f"Starting (full) joint training")

        if not args.skip_warmup:
            # Load best warmed-up model and initialize optimizer/scheduler
            logger.info("Loading best warmed-up encoder")
            args.load_dir = save_path
            pretrained_encoder, tokenizer, _ = load_encoder(device, args)
            # Train a copy of it
            logger.info("Copying target encoder")
            target_encoder = copy.deepcopy(pretrained_encoder)

        # Initialize optimizer & scheduler
        optimizer, scheduler = get_optimizer_scheduler(target_encoder, args, len(train_qa_pairs[1]),
                                                       len(dev_qa_pairs[1]),
                                                       args.num_train_epochs)

        # Train arguments
        args.per_gpu_train_batch_size = int(args.per_gpu_train_batch_size / args.gradient_accumulation_steps)
        # TODO: Run dev eval to compute a pre-trained best_acc
        best_acc = -1000.0
        best_epoch = -1

        # Run joint training for SUP sentence retrieval + answer phrase retrieval
        for ep_idx in range(int(args.num_train_epochs)):
            total_loss = 0.0

            # Get questions and corresponding answers with other metadata
            q_ids, questions, answers, titles, final_answers, final_titles = shuffle_data(train_qa_pairs)

            # Progress bar
            pbar = tqdm(get_top_phrases(
                mips, q_ids, questions, answers, titles, pretrained_encoder, tokenizer,  # encoder updated every epoch
                args.per_gpu_train_batch_size, args, final_answers, final_titles, agg_strat=args.warmup_agg_strat)
            )
            for step_idx, (q_ids, questions, answers, titles, outs, final_answers, final_titles) in enumerate(pbar):
                # INFO: `outs` contains topk phrases for each query

                train_dataloader, _, _ = get_question_dataloader(
                    questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
                )
                svs, evs, tgts, p_tgts = annotate_phrase_vecs(mips, q_ids, questions, answers, titles, outs, args,
                                                              label_strat=args.warmup_label_strat)

                target_encoder.train()
                # Start vectors
                svs_t = torch.Tensor(svs).to(device)  # shape: (bs, 2*topk, hid_dim)
                # End vectors
                evs_t = torch.Tensor(evs).to(device)
                # Target indices for phrases
                tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts]
                # Target indices for doc
                p_tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in p_tgts]

                # Create updated queries for second-hop search
                upd_queries = []
                for i, q in enumerate(questions):
                    # TODO: Should p_tgts also be considered? What if no target exists -- skip or something else?
                    for t in tgts[i].numpy():
                        upd_q_id = q_ids[i] + f"_{t}"
                        upd_evidence = outs[i][t]['answer']
                        upd_evidence_title = outs[i][t]['title'][0]
                        upd_answer = final_answers[i]
                        upd_answer_title = final_titles[i]
                        upd_queries.append(
                            (upd_q_id, q, upd_evidence, upd_evidence_title, upd_answer, upd_answer_title)
                        )

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
                    )  # INFO: `accs` is the accuracy of the top-1 probability prediction being a valid gold target

                    # Optimize, get acc and report
                    if loss is not None:
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        # Joint training: 2nd stage
                        upd_total_loss = 0.
                        upd_q_ids, upd_questions, upd_evidences, upd_evidence_titles, upd_answers, upd_answer_titles = \
                            list(zip(*upd_queries))
                        upd_questions = [uq + " " + upd_evidences[i] for i, uq in enumerate(upd_questions)]
                        top_phrases_upd = get_top_phrases(
                            mips, upd_q_ids, upd_questions, upd_evidences, upd_evidence_titles, pretrained_encoder,
                            tokenizer, args.per_gpu_train_batch_size, args, upd_answers, upd_answer_titles,
                            agg_strat=args.agg_strat
                        )
                        for upd_step_idx, (
                                upd_q_ids, upd_questions, upd_evidences, upd_evidence_titles, upd_outs, upd_answers,
                                upd_answer_titles) in enumerate(top_phrases_upd):
                            pbar.set_description(
                                f"2nd hop: {upd_step_idx + 1} / {len(top_phrases_upd)}"
                            )
                            upd_train_dataloader, _, _ = get_question_dataloader(
                                upd_questions, tokenizer, args.max_query_length,
                                batch_size=args.per_gpu_train_batch_size
                            )
                            u_svs, u_evs, u_tgts, u_p_tgts = annotate_phrase_vecs(mips, upd_q_ids, upd_questions,
                                                                                  upd_answers, upd_answer_titles,
                                                                                  upd_outs,
                                                                                  args,
                                                                                  label_strat=args.label_strat)
                            # Start vectors
                            u_svs_t = torch.Tensor(u_svs).to(device)  # shape: (bs, 2*topk, hid_dim)
                            # End vectors
                            u_evs_t = torch.Tensor(u_evs).to(device)
                            # Target indices for phrases
                            u_tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in
                                      u_tgts]
                            # Target indices for doc
                            u_p_tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in
                                        u_p_tgts]
                            # Train query encoder
                            assert len(upd_train_dataloader) == 1
                            for u_batch in upd_train_dataloader:
                                u_batch = tuple(t.to(device) for t in u_batch)
                                u_loss, u_accs = target_encoder.train_query(
                                    input_ids_=u_batch[0], attention_mask_=u_batch[1], token_type_ids_=u_batch[2],
                                    start_vecs=u_svs_t,
                                    end_vecs=u_evs_t,
                                    targets=u_tgts_t,
                                    p_targets=u_p_tgts_t,
                                )
                                if u_loss is not None:
                                    if args.gradient_accumulation_steps > 1:
                                        u_loss = u_loss / args.gradient_accumulation_steps
                                    upd_total_loss += u_loss.mean()
                        upd_total_loss /= (upd_step_idx+1)

                        # Combine first- and second-hop loss values
                        loss = 0.5*loss + 0.5*upd_total_loss

                        # Backpropagate combined loss and update model
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

                    # TODO: Figure out training acc. logging
                    # if accs is not None:
                    #     total_accs += accs
                    #     total_accs_k += [len(tgt) > 0 for tgt in tgts_t]
                    # else:
                    #     total_accs += [0.0] * len(tgts_t)
                    #     total_accs_k += [0.0] * len(tgts_t)
            step_idx += 1
            logger.info(
                f"Avg train loss ({step_idx} iterations): {total_loss / step_idx:.2f} | train "
                # +
                # f"acc@1: {sum(total_accs) / len(total_accs):.3f} | " +
                # f"acc@{args.top_k}: {sum(total_accs_k) / len(total_accs_k):.3f}"
            )
            # TODO: Add dev evaluation
            # TODO: Add best model saving


def get_top_phrases(mips, q_ids, questions, answers, titles, query_encoder, tokenizer, batch_size, args, final_answers,
                    final_titles, agg_strat=None):
    # Search
    step = batch_size
    search_fn = mips.search
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )
    for q_idx in tqdm(range(0, len(questions), step)):
        outs = query2vec(questions[q_idx:q_idx + step])  # outs[0] contains (start_vec_list, end_vec_list, query_tokens)
        start = np.concatenate([out[0] for out in outs], 0)
        end = np.concatenate([out[1] for out in outs], 0)
        query_vec = np.concatenate([start, end], 1)

        agg_strat = agg_strat if agg_strat is not None else args.agg_strat
        # For multi-hop warmup training
        # return_sent = agg_strat == "opt2a"

        outs = search_fn(
            query_vec,
            q_texts=questions[q_idx:q_idx + step], nprobe=args.nprobe,
            top_k=args.top_k, return_idxs=True,
            max_answer_length=args.max_answer_length, aggregate=args.aggregate, agg_strat=args.agg_strat,
            return_sent=True
        )

        yield (
            q_ids[q_idx:q_idx + step], questions[q_idx:q_idx + step], answers[q_idx:q_idx + step],
            titles[q_idx:q_idx + step], outs, final_answers[q_idx:q_idx + step], final_titles[q_idx:q_idx + step]
        )


def annotate_phrase_vecs(mips, q_ids, questions, answers, titles, phrase_groups, args, label_strat=None):
    assert mips is not None
    batch_size = len(answers)
    label_strat = label_strat if label_strat is not None else args.label_strat
    # Phrase groups are in size of (batch, top_k, emb_size)

    dummy_group = {
        'doc_idx': -1,
        'start_idx': 0, 'end_idx': 0,
        'answer': '',
        'start_vec': np.zeros(768),
        'end_vec': np.zeros(768),
        'context': '',
        'title': ['']
    }

    # Pad phrase groups (two separate top-k coming from start/end, so pad with top_k*2)
    for b_idx in range(len(phrase_groups)):
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

    # Annotate for L_phrase
    if 'phrase' in label_strat.split(','):
        match_fns = [
            drqa_regex_match_score if args.regex or ('trec' in q_id.lower()) else drqa_exact_match_score for q_id in
            q_ids
        ]
        targets = [
            [drqa_metric_max_over_ground_truths(match_fn, phrase['answer'], answer_set)
             for phrase in phrase_group]  # phrase_group is the top-k predictions
            for phrase_group, answer_set, match_fn in zip(phrase_groups, answers, match_fns)
        ]
        targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]

    # Annotate for L_sent (added for multi-hop warmup)
    # Checking if the predicted phrase is a substring of the gold annotated sentences or not
    if 'sent' in label_strat.split(','):
        match_fns = [drqa_substr_match_score for _ in q_ids]
        targets = [
            [drqa_metric_max_over_ground_truths(match_fn, phrase['answer'], answer_set)
             for phrase in phrase_group]  # phrase_group is the top-k predictions
            for phrase_group, answer_set, match_fn in zip(phrase_groups, answers, match_fns)
        ]
        targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]

    # Checking if the predicted phrase is a substr of the gold annotated sentence AND
    # if the token F1 score of the predicted evidence with the gold annotated sentence is >= a specified threshold
    if 'sent2' in label_strat.split(','):
        match_fns = [drqa_substr_f1_match_score for _ in q_ids]
        targets = [
            [drqa_metric_max_over_ground_truths(match_fn, {
                'pred_substr': phrase['answer'],
                'pred_f1': phrase['context'],
                'f1_threshold': args.evidence_f1_threshold,
            }, answer_set) for phrase in phrase_group]  # phrase_group is the top-k predictions
            for phrase_group, answer_set, match_fn in zip(phrase_groups, answers, match_fns)
        ]
        targets = [[ii if val else None for ii, val in enumerate(target)] for target in targets]

    # Annotate for L_doc
    if 'doc' in label_strat.split(','):
        p_targets = [
            [any(phrase['title'][0].lower() == tit.lower() for tit in title) for phrase in phrase_group]
            for phrase_group, title in zip(phrase_groups, titles)
        ]
        p_targets = [[ii if val else None for ii, val in enumerate(target)] for target in p_targets]

    return start_vecs, end_vecs, targets, p_targets


if __name__ == '__main__':
    # Setup: Get arguments, init logger, create output dir
    args, save_path = setup_run()

    paths_to_eval = {'train': args.train_path, 'dev': args.dev_path, 'test': args.test_path}
    if args.run_mode == 'train_query':
        mips = load_phrase_index(args)
        dev_init_em = None
        if not args.skip_init_eval:
            # Initial eval
            paths = paths_to_eval if args.eval_all_splits else {'dev': paths_to_eval['dev']}
            for split in paths:
                logger.info(f"Pre-training: Evaluating {args.load_dir} on the {split.upper()} set")
                print()
                # TODO: Implement `multihop`
                res = evaluate(args, mips, firsthop=True, save_pred=True, pred_fname_suffix="pretrain",
                               data_path=paths_to_eval[split], save_path=save_path)
                if split == 'dev':
                    if args.warmup_dev_metric=='joint':
                        dev_init_em = res[2][0] # phr_substr_evid_f1_top1: joint metric @ 1 on the dev set
                    elif args.warmup_dev_metric=='evidence':
                        dev_init_em = res[1][1] # evid_f1_score_top1: evidence metric f1 @1 on the dev set
                    else:
                        dev_init_em = res[0][1] # f1_score_top1
                    # TODO: Currently hard-coded to the joint metric; should change with args.warmup_dev_metric (Done)
            print("\n\n")

        # Train
        logger.info(f"Starting training...")
        print()
        train_query_encoder(args, save_path, mips, init_dev_acc=dev_init_em)
        print("\n\n")

        if not args.skip_final_eval:
            # Eval on test set
            args.load_dir = save_path
            print()
            # args.top_k = 10
            paths = paths_to_eval if args.eval_all_splits else {'test': paths_to_eval['test']}
            for split in paths_to_eval:
                logger.info(f"Final: Evaluating {args.load_dir} on the {split.upper()} set")
                evaluate(args, mips, data_path=paths_to_eval[split], firsthop=True, save_pred=True, save_path=save_path)
    elif args.run_mode == 'eval':
        mips = load_phrase_index(args)
        # Eval on test set
        logger.info(f"Evaluating {args.load_dir} on the TEST set")
        print()
        evaluate(args, mips, firsthop=True, save_pred=True, save_path=save_path)
    else:
        raise NotImplementedError

    print(f"\nOutput directory: {save_path}\n")
