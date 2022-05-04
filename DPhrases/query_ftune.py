import torch
import os
import random
import numpy as np
import logging
import math
import copy
from tqdm import tqdm
from apex import amp
import wandb

from IPython import embed

from densephrases.utils.squad_utils import get_question_dataloader
from densephrases.utils.single_utils import load_encoder
from densephrases.utils.open_utils import load_phrase_index, get_query2vec, load_qa_pairs, shuffle_data
from densephrases.utils.eval_utils import drqa_exact_match_score, drqa_regex_match_score, \
    drqa_metric_max_over_ground_truths, drqa_substr_match_score, drqa_substr_f1_match_score, \
    normalize_answer
from eval_phrase_retrieval import evaluate, to_arr
from densephrases import Options

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


run_id = str(__import__('calendar').timegm(__import__('time').gmtime()))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_model(model, path):
    make_dir(path)
    model.save_pretrained(path)
    return path


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

    if args.wandb:
        wandb.init(project="query-fine-tuning", config=args, entity="daqua")

    return args, save_path


def is_train_param(name):
    if name.endswith(".embeddings.word_embeddings.weight"):
        logger.info(f'freezing {name}')
        return False
    return True


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

    if args.fp16:
        encoder, optimizer = amp.initialize(encoder, optimizer, opt_level=args.fp16_opt_level)

    return encoder, optimizer, scheduler


def train_query_encoder(args, save_path, mips=None, init_dev_acc=None):
    # Freeze one for MIPS
    device = 'cuda' if args.cuda else 'cpu'
    logger.info("Loading pretrained encoder: this one is for MIPS (fixed)")
    target_encoder, tokenizer, _ = load_encoder(device, args, query_only=True)

    # MIPS
    if mips is None:
        mips = load_phrase_index(args)

    # Load train and dev data
    train_qa_pairs = load_qa_pairs(args.train_path, args, multihop=True)
    dev_qa_pairs = load_qa_pairs(args.dev_path, args, multihop=True)

    # Train arguments
    args.per_gpu_train_batch_size = int(args.per_gpu_train_batch_size / args.gradient_accumulation_steps)
    best_acc = -1000.0 if init_dev_acc is None else init_dev_acc
    best_epoch = -1

    if not args.skip_warmup:
        logger.info(f"Starting warmup training")
        # Initialize optimizer & scheduler
        target_encoder, optimizer, scheduler = get_optimizer_scheduler(target_encoder, args, len(train_qa_pairs[1]),
                                                                       len(dev_qa_pairs[1]),
                                                                       args.num_firsthop_epochs)

        # Initially, save pre-trained model as the best model
        save_model(target_encoder, save_path)

        # Warm-up the model with a first-hop retrieval objective
        for ep_idx in range(int(args.num_firsthop_epochs)):
            # Training
            total_loss = 0.
            total_accs = []
            total_accs_k = []
            backprop_steps = 0.

            # Load training dataset
            q_ids, levels, questions, answers, titles, final_answers, final_titles = shuffle_data(train_qa_pairs, args)

            # Progress bar
            pbar = tqdm(get_top_phrases(
                mips, q_ids, levels, questions, answers, titles, target_encoder, tokenizer,  # encoder updated every epoch
                args.per_gpu_train_batch_size, args, final_answers, final_titles, agg_strat=args.warmup_agg_strat,
                always_return_sent=True)
            )

            for step_idx, (q_ids, levels, questions, answers, titles, outs, final_answers, final_titles) in enumerate(pbar):
                # INFO: `outs` contains topk phrases for each query

                train_dataloader, _, _ = get_question_dataloader(
                    questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
                )
                svs, evs, tgts, p_tgts = annotate_phrase_vecs(mips, q_ids, questions, to_arr(answers, 2),
                                                              to_arr(titles, 2), outs, args,
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
                        backprop_steps += 1
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        total_loss += loss.item()
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), args.max_grad_norm)

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        target_encoder.zero_grad()

                        total_accs += accs
                        total_accs_k += [len(tgt) > 0 for tgt in tgts_t]
                    else:
                        total_accs += [0.0] * len(tgts_t)
                        total_accs_k += [0.0] * len(tgts_t)

                    pbar.set_description(
                        f"Ep {ep_idx + 1} Tr loss: " + ("skipped, " if loss is None else f"{loss.item():.2f}, ") +
                        f"acc: {np.mean(total_accs[-1]):.3f}"
                    )

                if args.wandb:
                    wandb.log({
                        "warmup_train/epoch": ep_idx + 1,
                        "warmup_train/loss": float('inf') if loss is None else loss.item(),
                        "warmup_train/avg_total_loss": total_loss / max(backprop_steps, 1),
                        "warmup_train/acc@1": float('inf') if loss is None else np.mean(total_accs[-1]),
                        f"warmup_train/acc@{args.top_k}": float('inf') if loss is None else np.mean(total_accs_k[-1]),
                    })

            logger.info(
                f"Avg train loss ({step_idx + 1} steps, {backprop_steps} backprop steps): {total_loss / max(backprop_steps, 1):.2f} | train " +
                f"acc@1: {np.mean(total_accs):.3f} | acc@{args.top_k}: {np.mean(total_accs_k):.3f}"
            )

            # Save epoch model
            last_saved_path = save_model(target_encoder, os.path.join(save_path, f"warmup_ep{ep_idx + 1}"))

            if not args.skip_warmup_dev_eval:
                # Dev evaluation
                logger.info("Evaluating on the DEV set")
                new_args = copy.deepcopy(args)
                new_args.top_k = 10
                new_args.test_path = args.dev_path
                new_args.load_dir = last_saved_path
                eval_res = evaluate(new_args, mips=mips, tokenizer=tokenizer, firsthop=True,
                                    pred_fname_suffix=f"warmup_ep{ep_idx + 1}", save_path=save_path,
                                    save_pred=True, always_return_sent=True, agg_strat=args.warmup_agg_strat)
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

                if args.wandb:
                    wandb.log({
                        "warmup_dev_eval/epoch": ep_idx + 1,
                        "warmup_dev_eval/acc@1": dev_em,
                        "warmup_dev_eval/f1@1": dev_f1,
                        f"warmup_dev_eval/acc@{new_args.top_k}": dev_emk,
                        f"warmup_dev_eval/f1@{new_args.top_k}": dev_f1k,
                        "warmup_dev_eval/evidence_acc@1": evid_dev_em,
                        "warmup_dev_eval/evidence_f1@1": evid_dev_f1,
                        f"warmup_dev_eval/evidence_acc@{new_args.top_k}": evid_dev_emk,
                        f"warmup_dev_eval/evidence_f1@{new_args.top_k}": evid_dev_f1k,
                        "warmup_dev_eval/joint_substr_f1@1": joint_substr_f1,
                        f"warmup_dev_eval/joint_substr_f1@{new_args.top_k}": joint_substr_f1k
                    })

                # Save best model
                if dev_metrics[args.warmup_dev_metric] > best_acc:
                    best_acc = dev_metrics[args.warmup_dev_metric]
                    best_epoch = ep_idx + 1
                    save_model(target_encoder, save_path)
                    logger.info(f"Saved best warmup model with ({args.warmup_dev_metric}) acc. {best_acc:.3f} into {save_path}")
            else:
                save_model(target_encoder, save_path)
        print()
        if best_acc > 0:
            logger.info(f"Best model (epoch {best_epoch}) with accuracy {best_acc:.3f} saved at {save_path}")
            if args.wandb:
                wandb.run.summary["best_accuracy_warmup"] = best_acc
                wandb.run.summary["best_epoch_warmup"] = best_epoch
                wandb.run.summary["best_model_warmup"] = os.path.join(save_path, f"warmup_ep{best_epoch}")

    if not args.warmup_only:
        logger.info(f"Starting (full) joint training")

        if not args.skip_warmup:
            # Load best warmed-up model and initialize optimizer/scheduler
            logger.info("Loading best warmed-up encoder")
            args.load_dir = save_path
            target_encoder, tokenizer, _ = load_encoder(device, args, query_only=True)
            best_acc = -1000.0
            best_epoch = -1
            # If warmup was executed, run dev eval before starting joint training
            new_args = copy.deepcopy(args)
            new_args.top_k = 10
            new_args.test_path = args.dev_path
            dev_em, dev_f1, dev_emk, dev_f1k = evaluate(new_args, mips=mips, tokenizer=tokenizer, multihop=True,
                                                        pred_fname_suffix="joint_ep0", save_path=save_path,
                                                        save_pred=True, always_return_sent=True,
                                                        agg_strat=(args.warmup_agg_strat, args.agg_strat))
            # Dev metric to use for picking the best epoch
            dev_metrics = {
                "phrase": dev_em,
                "phrase_k": dev_emk,
            }
            if dev_metrics[args.dev_metric] > best_acc:
                best_acc = dev_metrics[args.dev_metric]
                best_epoch = 0
            if args.wandb:
                wandb.log({
                    "joint_dev_eval/epoch": 0,
                    "joint_dev_eval/acc@1": dev_em,
                    "joint_dev_eval/f1@1": dev_f1,
                    f"joint_dev_eval/acc@{new_args.top_k}": dev_emk,
                    f"joint_dev_eval/f1@{new_args.top_k}": dev_f1k
                })
        else:
            # Initially, save pre-trained model as the best model
            save_model(target_encoder, save_path)

        # Initialize optimizer & scheduler
        target_encoder, optimizer, scheduler = get_optimizer_scheduler(target_encoder, args, len(train_qa_pairs[1]),
                                                                       len(dev_qa_pairs[1]),
                                                                       args.num_train_epochs)
        # First-hop weight for convex combination of first- and second-hop loss values
        lmbda = args.joint_loss_lambda
        assert 0 <= lmbda <= 1

        # Run joint training for SUP sentence retrieval + answer phrase retrieval
        for ep_idx in range(int(args.num_train_epochs)):
            total_loss = 0.0
            total_accs = []
            total_accs_k = []
            total_u_accs = []
            total_u_accs_k = []
            backprop_steps = 0.

            # Get questions and corresponding answers with other metadata
            q_ids, levels, questions, answers, titles, final_answers, final_titles = shuffle_data(train_qa_pairs, args)

            # Joint training: 1st stage
            pbar = tqdm(get_top_phrases(
                mips, q_ids, levels, questions, answers, titles, target_encoder, tokenizer,  # encoder updated every epoch
                args.per_gpu_train_batch_size, args, final_answers, final_titles, agg_strat=args.warmup_agg_strat,
                always_return_sent=True)
            )
            for step_idx, (q_ids, levels, questions, answers, titles, outs, final_answers, final_titles) in enumerate(pbar):
                # INFO: `outs` contains topk phrases for each query
                n_hop1_skipped_ans = 0.  # Track how many first hops skipped because of no valid ans retrieval
                n_hop1_skipped_easy = 0.  # Track how many first hops skipped because question was "easy"
                n_hop2_skipped = 0.  # Track how many second hops skipped because of no valid ans retrieval
                fhop_loss, ans_loss, loss = None, None, None

                train_dataloader, _, _ = get_question_dataloader(
                    questions, tokenizer, args.max_query_length, batch_size=args.per_gpu_train_batch_size
                )
                svs, evs, tgts, p_tgts = annotate_phrase_vecs(mips, q_ids, questions, to_arr(answers, 2),
                                                              to_arr(titles, 2), outs, args,
                                                              label_strat=args.warmup_label_strat)

                target_encoder.train()
                # Start vectors
                svs_t = torch.Tensor(svs).to(device)  # shape: (bs, 2*topk, hid_dim)
                # End vectors
                evs_t = torch.Tensor(evs).to(device)

                if not args.skip_first_hop:
                    # Update tgt and p_tgt to filter out loss contributions from easy questions 
                    tgts = [[None] * len(tgt) if lev == 'easy' else tgt for tgt, lev in zip(tgts, levels)]
                    p_tgts = [[None] * len(p_tgt) if lev == 'easy' else p_tgt for p_tgt, lev in zip(p_tgts, levels)]
                else:
                    # Update tgt and p_tgt to filter out first hop loss contributions from all questions 
                    tgts = [[None] * len(tgt) for tgt in tgts]
                    p_tgts = [[None] * len(p_tgt) for p_tgt in p_tgts]

                # Target indices for phrases
                tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in tgts]
                # Target indices for doc
                p_tgts_t = [torch.Tensor([tgt_ for tgt_ in tgt if tgt_ is not None]).to(device) for tgt in p_tgts]

                # Create updated queries for second-hop search using filtered phrases in tgts
                upd_queries = []
                for i, q in enumerate(questions):
                    # Skip "yes/no" questions
                    if normalize_answer(final_answers[i]) in ['yes', 'no']:
                        continue
                    # Retain the original query text for "easy" questions (i.e. store empty evidence)
                    if levels[i] == 'easy' or args.skip_first_hop:
                        if not args.skip_first_hop:
                            n_hop1_skipped_easy += 1
                        upd_q_id = q_ids[i] + "_uc"
                        upd_level = levels[i]
                        upd_evidence = ''
                        upd_evidence_title = ''
                        upd_answer = final_answers[i]
                        upd_answer_title = final_titles[i]
                        upd_queries.append(
                            (upd_q_id, upd_level, q, upd_evidence, upd_evidence_title, upd_answer, upd_answer_title)
                        )
                    else:
                        # Track first-hop evidence for non-"easy" questions
                        if len(tgts_t[i]) == 0 and len(p_tgts_t[i]) == 0:
                            n_hop1_skipped_ans += 1
                        for t in tgts_t[i][:args.top_k]:
                            t = int(t.item())
                            upd_q_id = q_ids[i] + f"_{t}"
                            upd_level = levels[i]
                            if not args.upd_sent_evd:
                                upd_evidence = outs[i][t]['answer']
                            elif args.upd_sent_evd:
                                upd_evidence = outs[i][t]['context']
                            upd_evidence_title = outs[i][t]['title'][0]
                            upd_answer = final_answers[i]
                            upd_answer_title = final_titles[i]
                            upd_queries.append(
                                (upd_q_id, upd_level, q, upd_evidence, upd_evidence_title, upd_answer, upd_answer_title)
                            )
                n_hop1_skipped_ans /= len(questions)
                n_hop1_skipped_easy /= len(questions)

                # Compute first-hop loss
                assert len(train_dataloader) == 1
                for batch in train_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    fhop_loss, accs = target_encoder.train_query(
                        input_ids_=batch[0], attention_mask_=batch[1], token_type_ids_=batch[2],
                        start_vecs=svs_t,
                        end_vecs=evs_t,
                        targets=tgts_t,
                        p_targets=p_tgts_t,
                    )
                    if fhop_loss is not None:
                        if args.gradient_accumulation_steps > 1:
                            fhop_loss = fhop_loss / args.gradient_accumulation_steps
                        total_accs += accs
                        total_accs_k += [len(tgt) > 0 for tgt in tgts_t]
                    else:
                        total_accs += [0.0] * len(tgts_t)
                        total_accs_k += [0.0] * len(tgts_t)

                # Joint training: 2nd stage
                total_u_accs_step = []
                total_u_accs_k_step = []
                # Skip 2nd-hop if no updated queries constructed
                if len(upd_queries) > 0:
                    upd_q_ids, upd_levels, upd_questions, upd_evidences, upd_evidence_titles, \
                    upd_answers, upd_answer_titles = list(zip(*upd_queries))
                    upd_questions = [uq + " " + upd_evidences[i] for i, uq in enumerate(upd_questions)]
                    top_phrases_upd = get_top_phrases(
                        mips, upd_q_ids, upd_levels, upd_questions, upd_evidences, upd_evidence_titles,
                        target_encoder, tokenizer, args.per_gpu_train_batch_size, args, upd_answers,
                        upd_answer_titles, agg_strat=args.agg_strat, always_return_sent=True, silent=True
                    )
                    u_loss_arr = []
                    for upd_step_idx, (
                            upd_q_ids, upd_levels, upd_questions, upd_evidences, upd_evidence_titles, upd_outs, upd_answers,
                            upd_answer_titles) in enumerate(top_phrases_upd):
                        upd_train_dataloader, _, _ = get_question_dataloader(
                            upd_questions, tokenizer, args.max_query_length,
                            batch_size=args.per_gpu_train_batch_size
                        )
                        u_svs, u_evs, u_tgts, u_p_tgts = annotate_phrase_vecs(mips, upd_q_ids, upd_questions,
                                                                              to_arr(upd_answers, 2),
                                                                              to_arr(upd_answer_titles, 2),
                                                                              upd_outs, args,
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
                        # Compute partial second-hop loss and store in u_loss_arr
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
                                u_loss_arr.append(u_loss)
                                total_u_accs += u_accs
                                total_u_accs_step += u_accs
                                total_u_accs_k += [len(tgt) > 0 for tgt in u_tgts_t]
                                total_u_accs_k_step += [len(tgt) > 0 for tgt in u_tgts_t]
                                # Calculate number of upd_queries with no targets in this batch
                                no_tgt = 0.
                                for i in range(len(u_tgts_t)):
                                    no_tgt += len(u_tgts_t[i]) == 0 and len(u_p_tgts_t[i]) == 0
                                no_tgt /= len(u_tgts_t)
                                n_hop2_skipped += no_tgt
                            else:
                                n_hop2_skipped += 1
                                total_u_accs += [0.0] * len(u_tgts_t)
                                total_u_accs_step += [0.0] * len(u_tgts_t)
                                total_u_accs_k += [0.0] * len(u_tgts_t)
                                total_u_accs_k_step += [0.0] * len(u_tgts_t)
                    n_hop2_skipped /= (upd_step_idx + 1)
                    if len(u_loss_arr) == 0:
                        n_hop2_skipped = 1
                        if fhop_loss is not None:
                            loss = lmbda * fhop_loss
                    else:
                        # Average ans_loss over total number of original questions
                        ans_loss = sum(u_loss_arr) / len(u_loss_arr)
                        # Final loss: combine first- and second-hop loss values
                        if fhop_loss is not None:
                            loss = lmbda * fhop_loss + (1 - lmbda) * ans_loss
                        else:
                            loss = (1 - lmbda) * ans_loss
                else:
                    n_hop2_skipped = 1
                    if fhop_loss is not None:
                        loss = lmbda * fhop_loss

                if type(loss) is torch.Tensor:  # Not checking "if None" since loss could also just be a 0
                    backprop_steps += 1
                    # Backpropagate combined loss and update model
                    try:
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                    except:
                        logger.info("ERROR: loss.backward()")
                        logger.info(f"Number of updated queries: {len(upd_queries)}")
                        raise ValueError("loss.backward()")

                    total_loss += loss.item()
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(target_encoder.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    target_encoder.zero_grad()

                    pbar.set_description(
                        f"Ep{ep_idx + 1}_step{step_idx + 1}: loss: {loss.item():.2f}, " +
                        f"acc1: " + ("skipped, " if fhop_loss is None else f"{np.mean(total_accs[-1]):.3f}, ") +
                        "acc2: " + ("skipped" if n_hop2_skipped == 1 else f"{np.mean(total_u_accs_step):.3f}")
                    )
                else:
                    pbar.set_description(f"Ep{ep_idx + 1}_step{step_idx + 1}: loss: skipped")

                if args.wandb:
                    wandb.log({
                        "joint_train/epoch": ep_idx + 1,
                        "joint_train/first_hop_loss": float('inf') if fhop_loss is None else fhop_loss.item(),
                        "joint_train/second_hop_loss": float('inf') if n_hop2_skipped == 1 else ans_loss.item(),
                        "joint_train/joint_loss": float('inf') if type(loss) is not torch.Tensor else loss.item(),
                        "joint_train/avg_total_loss": float('inf') if type(loss) is not torch.Tensor else total_loss / max(backprop_steps, 1),
                        "joint_train/first_hop_acc@1": float('inf') if fhop_loss is None else np.mean(total_accs[-1]),
                        f"joint_train/first_hop_acc@{args.top_k}": float('inf') if fhop_loss is None else np.mean(total_accs_k[-1]),
                        "joint_train/second_hop_acc@1": float('inf') if n_hop2_skipped == 1 else np.mean(total_u_accs_step),
                        f"joint_train/second_hop_acc@{args.top_k}": float('inf') if n_hop2_skipped == 1 else np.mean(total_u_accs_k_step),
                        f"joint_train/first_hop_skipped_easy": n_hop1_skipped_easy,
                        f"joint_train/first_hop_skipped_ans": n_hop1_skipped_ans,
                        f"joint_train/second_hop_skipped_ans": n_hop2_skipped,
                    })

            logger.info(
                f"Avg train loss ({step_idx + 1} steps, {backprop_steps} backprop steps): {total_loss / max(backprop_steps, 1):.3f} | "
                +
                f"(first-hop) acc@1: {np.mean(total_accs):.3f}, " +
                f"acc@{args.top_k}: {np.mean(total_accs_k):.3f} "
                +
                f"| (second-hop) acc@1: {np.mean(total_u_accs):.3f}, " +
                f"acc@{args.top_k}: {np.mean(total_u_accs_k):.3f} "
            )

            # Save epoch model
            last_saved_path = save_model(target_encoder, os.path.join(save_path, f"joint_ep{ep_idx + 1}"))

            # Dev evaluation
            logger.info("Evaluating on the DEV set")
            new_args = copy.deepcopy(args)
            new_args.top_k = 10
            new_args.test_path = args.dev_path
            new_args.load_dir = last_saved_path
            dev_em, dev_f1, dev_emk, dev_f1k = evaluate(new_args, mips=mips, tokenizer=tokenizer, multihop=True,
                                                        pred_fname_suffix=f"joint_ep{ep_idx + 1}", save_path=save_path,
                                                        save_pred=True, always_return_sent=True,
                                                        agg_strat=(args.warmup_agg_strat, args.agg_strat))
            # Warm-up dev metric to use for picking the best epoch
            dev_metrics = {
                "phrase": dev_em,
                "phrase_k": dev_emk,
                # TODO: Add any other dev metrics for joint training?
            }
            logger.info(f"Dev set acc@1: {dev_em:.3f}, f1@1: {dev_f1:.3f}")

            if args.wandb:
                wandb.log({
                    "joint_dev_eval/epoch": ep_idx + 1,
                    "joint_dev_eval/acc@1": dev_em,
                    "joint_dev_eval/f1@1": dev_f1,
                    f"joint_dev_eval/acc@{new_args.top_k}": dev_emk,
                    f"joint_dev_eval/f1@{new_args.top_k}": dev_f1k
                })

            # Save best model
            if dev_metrics[args.dev_metric] > best_acc:
                best_acc = dev_metrics[args.dev_metric]
                best_epoch = ep_idx + 1
                save_model(target_encoder, save_path)
                logger.info(f"Saved best model with ({args.dev_metric}) acc. {best_acc:.3f} into {save_path}")
                if args.wandb:
                    wandb.run.summary["best_accuracy_joint"] = best_acc
                    wandb.run.summary["best_epoch_joint"] = best_epoch
                    wandb.run.summary["best_model_joint"] = save_path
        print()
        logger.info(f"Best model (epoch {best_epoch}) with accuracy {best_acc:.3f} saved at {save_path}")


def get_top_phrases(mips, q_ids, levels, questions, answers, titles, query_encoder, tokenizer, batch_size, args,
                    final_answers, final_titles, agg_strat=None, always_return_sent=False, silent=False):
    # Search
    step = batch_size
    search_fn = mips.search
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size, silent=True
    )
    iterator = range(0, len(questions), step)
    if not silent:
        iterator = tqdm(iterator)
    for q_idx in iterator:
        outs = query2vec(questions[q_idx:q_idx + step])  # outs[0] contains (start_vec_list, end_vec_list, query_tokens)
        start = np.concatenate([out[0] for out in outs], 0)
        end = np.concatenate([out[1] for out in outs], 0)
        query_vec = np.concatenate([start, end], 1)

        agg_strat = agg_strat if agg_strat is not None else args.agg_strat
        # For multi-hop warmup training
        return_sent = always_return_sent or agg_strat == "opt2a"

        outs = search_fn(
            query_vec,
            q_texts=questions[q_idx:q_idx + step], nprobe=args.nprobe,
            top_k=args.top_k, return_idxs=True,
            max_answer_length=args.max_answer_length, aggregate=args.aggregate, agg_strat=agg_strat,
            return_sent=return_sent
        )

        yield (
            q_ids[q_idx:q_idx + step], levels[q_idx:q_idx + step], questions[q_idx:q_idx + step],
            answers[q_idx:q_idx + step], titles[q_idx:q_idx + step], outs, final_answers[q_idx:q_idx + step],
            final_titles[q_idx:q_idx + step]
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
                'pred_f1': phrase['context'],  # sentence of the predicted answer phrase
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
        init_dev_acc = None

        if not args.skip_init_eval:
            # Initial eval
            paths = paths_to_eval if args.eval_all_splits else {'dev': paths_to_eval['dev']}
            for split in paths:
                logger.info(f"Pre-training: Evaluating {args.load_dir} on the {split.upper()} set")
                print()
                if not args.skip_warmup:
                    # Evaluate warmup (first-hop) stage
                    res = evaluate(args, mips, firsthop=True, save_pred=True, pred_fname_suffix="warmup_ep0",
                                   data_path=paths_to_eval[split], save_path=save_path, always_return_sent=True,
                                   agg_strat=args.warmup_agg_strat)
                    if split == 'dev':
                        dev_metrics = {
                            "phrase": res[0][0],  # phr_em_top1: phrase metric @1 on the dev set
                            "evidence": res[1][1],  # evid_f1_top1: evidence metric @1 on the dev set
                            "joint": res[2][0]  # phr_substr_evid_f1_top1: joint metric @ 1 on the dev set
                        }
                        init_dev_acc = dev_metrics[args.warmup_dev_metric]
                        if args.wandb:
                            dev_em, dev_f1, dev_emk, dev_f1k = res[0]
                            evid_dev_em, evid_dev_f1, evid_dev_emk, evid_dev_f1k = res[1]
                            joint_substr_f1, joint_substr_f1k = res[2]
                            wandb.log({
                                "warmup_dev_eval/epoch": 0,
                                "warmup_dev_eval/acc@1": dev_em,
                                "warmup_dev_eval/f1@1": dev_f1,
                                f"warmup_dev_eval/acc@{args.top_k}": dev_emk,
                                f"warmup_dev_eval/f1@{args.top_k}": dev_f1k,
                                "warmup_dev_eval/evidence_acc@1": evid_dev_em,
                                "warmup_dev_eval/evidence_f1@1": evid_dev_f1,
                                f"warmup_dev_eval/evidence_acc@{args.top_k}": evid_dev_emk,
                                f"warmup_dev_eval/evidence_f1@{args.top_k}": evid_dev_f1k,
                                "warmup_dev_eval/joint_substr_f1@1": joint_substr_f1,
                                f"warmup_dev_eval/joint_substr_f1@{args.top_k}": joint_substr_f1k
                            })
                else:
                    # Evaluate joint (multi-hop) stage
                    res = evaluate(args, mips, multihop=True, save_pred=True, pred_fname_suffix="joint_ep0",
                                   data_path=paths_to_eval[split], save_path=save_path, always_return_sent=True,
                                   agg_strat=(args.warmup_agg_strat, args.agg_strat))
                    # If skipping warmup training, initialize dev acc with the joint eval
                    if split == 'dev':
                        dev_metrics = {
                            "phrase": res[0],  # phr_em_top1: phrase metric @1 on the dev set
                            "phrase_k": res[2],  # phr_em_topk: phrase metric @k on the dev set
                        }
                        init_dev_acc = dev_metrics[args.dev_metric]
                        if args.wandb:
                            dev_em, dev_f1, dev_emk, dev_f1k = res
                            wandb.log({
                                "joint_dev_eval/epoch": 0,
                                "joint_dev_eval/acc@1": dev_em,
                                "joint_dev_eval/f1@1": dev_f1,
                                f"joint_dev_eval/acc@{args.top_k}": dev_emk,
                                f"joint_dev_eval/f1@{args.top_k}": dev_f1k
                            })
                print("\n\n")

        # Train
        logger.info(f"Starting training...")
        print()
        train_query_encoder(args, save_path, mips, init_dev_acc=init_dev_acc)
        print("\n\n")

        if not args.skip_final_eval:
            # Eval on test set
            args.load_dir = save_path
            print()
            paths = paths_to_eval if args.eval_all_splits else {'test': paths_to_eval['test']}
            for split in paths:
                logger.info(f"Final: Evaluating {args.load_dir} on the {split.upper()} set")
                res = evaluate(args, mips, data_path=paths_to_eval[split], firsthop=args.warmup_only,
                               multihop=(not args.warmup_only), save_pred=True, save_path=save_path,
                               always_return_sent=True, agg_strat=(args.warmup_agg_strat
                               if args.warmup_only else (args.warmup_agg_strat, args.agg_strat)))
                if args.wandb:
                    wandb_panel = f"{'warmup' if args.warmup_only else 'joint'}_{split}_eval"
                    if args.warmup_only:
                        split_em, split_f1, split_emk, split_f1k = res[0]
                        evid_split_em, evid_split_f1, evid_split_emk, evid_split_f1k = res[1]
                        joint_substr_f1, joint_substr_f1k = res[2]
                        wandb.log({
                            f"{wandb_panel}/acc@1": split_em,
                            f"{wandb_panel}/f1@1": split_f1,
                            f"{wandb_panel}/acc@{args.top_k}": split_emk,
                            f"{wandb_panel}/f1@{args.top_k}": split_f1k,
                            f"{wandb_panel}/evidence_acc@1": evid_split_em,
                            f"{wandb_panel}/evidence_f1@1": evid_split_f1,
                            f"{wandb_panel}/evidence_acc@{args.top_k}": evid_split_emk,
                            f"{wandb_panel}/evidence_f1@{args.top_k}": evid_split_f1k,
                            f"{wandb_panel}/joint_substr_f1@1": joint_substr_f1,
                            f"{wandb_panel}/joint_substr_f1@{args.top_k}": joint_substr_f1k
                        })
                    else:
                        split_em, split_f1, split_emk, split_f1k = res
                        wandb.log({
                            f"{wandb_panel}/acc@1": split_em,
                            f"{wandb_panel}/f1@1": split_f1,
                            f"{wandb_panel}/acc@{args.top_k}": split_emk,
                            f"{wandb_panel}/f1@{args.top_k}": split_f1k,
                        })

    elif args.run_mode == 'eval':
        mips = load_phrase_index(args)
        paths = paths_to_eval if args.eval_all_splits else {'test': paths_to_eval['test']}
        for split in paths:
            if not args.ret_multi_stage_model:
                logger.info(f"Evaluating {args.load_dir} on the {split.upper()} set")
                res = evaluate(args, mips, data_path=paths_to_eval[split], firsthop=args.warmup_only,
                               multihop=(not args.warmup_only), save_pred=True, save_path=save_path,
                               always_return_sent=True)
            else:
                logger.info(f"Evaluation of {split.upper()} set done using warmup model on first stage and pretrained model on second stage")
                args.ret_both_warm_pretrain = True
                device = 'cuda' if args.cuda else 'cpu'
                ptrained_model, warmup_model, tokenizer_model, _ = load_encoder(device, args, query_only=True)
                res = evaluate(args, mips, (ptrained_model, warmup_model), tokenizer_model,
                               data_path=paths_to_eval[split], firsthop=args.warmup_only,
                               multihop=(not args.warmup_only), save_pred=True, save_path=save_path,
                               always_return_sent=True)

            if args.wandb:
                wandb_panel = f"{'warmup' if args.warmup_only else 'joint'}_{split}_eval"
                if args.warmup_only:
                    split_em, split_f1, split_emk, split_f1k = res[0]
                    evid_split_em, evid_split_f1, evid_split_emk, evid_split_f1k = res[1]
                    joint_substr_f1, joint_substr_f1k = res[2]
                    wandb.log({
                        f"{wandb_panel}/acc@1": split_em,
                        f"{wandb_panel}/f1@1": split_f1,
                        f"{wandb_panel}/acc@{args.top_k}": split_emk,
                        f"{wandb_panel}/f1@{args.top_k}": split_f1k,
                        f"{wandb_panel}/evidence_acc@1": evid_split_em,
                        f"{wandb_panel}/evidence_f1@1": evid_split_f1,
                        f"{wandb_panel}/evidence_acc@{args.top_k}": evid_split_emk,
                        f"{wandb_panel}/evidence_f1@{args.top_k}": evid_split_f1k,
                        f"{wandb_panel}/joint_substr_f1@1": joint_substr_f1,
                        f"{wandb_panel}/joint_substr_f1@{args.top_k}": joint_substr_f1k
                    })
                else:
                    split_em, split_f1, split_emk, split_f1k = res
                    wandb.log({
                        f"{wandb_panel}/acc@1": split_em,
                        f"{wandb_panel}/f1@1": split_f1,
                        f"{wandb_panel}/acc@{args.top_k}": split_emk,
                        f"{wandb_panel}/f1@{args.top_k}": split_f1k,
                    })
    else:
        raise NotImplementedError

    print(f"\nOutput directory: {save_path}\n")
