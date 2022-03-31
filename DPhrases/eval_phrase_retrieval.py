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
import csv
import subprocess

from time import time
from tqdm import tqdm

from densephrases.utils.eval_utils import normalize_answer, f1_score, exact_match_score, drqa_exact_match_score, \
    drqa_regex_match_score, drqa_metric_max_over_ground_truths, drqa_normalize, drqa_substr_match_score, \
    drqa_substr_f1_match_score
from densephrases.utils.single_utils import load_encoder
from densephrases.utils.open_utils import load_phrase_index, get_query2vec, load_qa_pairs
from densephrases.utils.kilt.eval import evaluate as kilt_evaluate
from densephrases.utils.kilt.kilt_utils import store_data as kilt_store_data
from densephrases import Options


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def embed_all_query(questions, args, query_encoder, tokenizer, batch_size=64):
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )

    all_outs = []
    for q_idx in tqdm(range(0, len(questions), batch_size)):
        outs = query2vec(questions[q_idx:q_idx+batch_size])
        all_outs += outs
    start = np.concatenate([out[0] for out in all_outs], 0)
    end = np.concatenate([out[1] for out in all_outs], 0)
    query_vec = np.concatenate([start, end], 1)
    logger.info(f'Query reps: {query_vec.shape}')
    return query_vec


def evaluate(args, mips=None, query_encoder=None, tokenizer=None, q_idx=None, firsthop=False, multihop=False,
             save_pred=None, pred_fname_suffix="", data_path=None, agg_strat=None, save_path=None):
    # Set path to load evaluation data
    data_path = data_path if data_path is not None else args.test_path
    # Set aggregation strategy
    agg_strat = agg_strat if agg_strat is not None else 'opt2a' if firsthop else args.agg_strat
    # If "opt2a", reduce the predicted context to the sentence in which the predicted phrase is found
    return_sent = agg_strat == "opt2a"

    # Load dataset and encode queries
    if firsthop or multihop:
        qids, questions, answers, titles, final_answers, final_titles = load_qa_pairs(data_path, args, q_idx,
                                                                                      multihop=True)
    else:
        qids, questions, answers, titles = load_qa_pairs(data_path, args, q_idx)

    if query_encoder is None:
        logger.info(f'Query encoder will be loaded from {args.load_dir}')
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer, _ = load_encoder(device, args)
    query_vec = embed_all_query(questions, args, query_encoder, tokenizer)

    # Load MIPS
    if mips is None:
        mips = load_phrase_index(args)

    # Search
    step = args.eval_batch_size
    logger.info(f'Aggregation strategy used: {agg_strat}')
    predictions = []
    evidences = []
    pred_titles = []
    scores = []
    se_poss = []
    for q_idx in tqdm(range(0, len(questions), step)):
        result = mips.search(
            query_vec[q_idx:q_idx+step],
            q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
            top_k=args.top_k, max_answer_length=args.max_answer_length,
            aggregate=args.aggregate, agg_strat=agg_strat, return_sent=return_sent
        )
        prediction = [[ret['answer'] for ret in out][:args.top_k] if len(out) > 0 else [''] for out in result]
        evidence = [[ret['context'] for ret in out][:args.top_k] if len(out) > 0 else [''] for out in result]
        title = [[ret['title'] for ret in out][:args.top_k] if len(out) > 0 else [['']] for out in result]
        score = [[ret['score'] for ret in out][:args.top_k] if len(out) > 0 else [-1e10] for out in result]
        se_pos = [[(ret['start_pos'], ret['end_pos']) for ret in out][:args.top_k] if len(out) > 0 else [(0,0)] for out in result]
        predictions += prediction
        evidences += evidence
        pred_titles += title
        scores += score
        se_poss += se_pos

    # logger.info(f"Avg. {sum(mips.num_docs_list)/len(mips.num_docs_list):.2f} number of docs per query")
    eval_fn = evaluate_results if not args.is_kilt else evaluate_results_kilt
    return eval_fn(predictions, qids, questions, answers, titles, args, evidences,
                   scores, pred_titles, se_positions=se_poss, firsthop=firsthop, multihop=multihop,
                   save_pred=save_pred, pred_fname_suffix=pred_fname_suffix, data_path=data_path,
                   save_path=save_path)


def evaluate_results(predictions, qids, questions, answers, titles, args, evidences,
                     scores, pred_titles, se_positions=None, firsthop=False, multihop=False,
                     save_pred=None, pred_fname_suffix="", data_path=None, save_path=None):
    """
    TODO: Implement logic for arg `multihop`
    TODO: Implement evaluation of `titles`
    """
    # Set path to load evaluation data
    data_path = data_path if data_path is not None else args.test_path

    # Filter if there's candidate
    if args.candidate_path is not None:
        candidates = set()
        with open(args.candidate_path) as f:
            for line in f:
                line = line.strip().lower()
                candidates.add(line)
        logger.info(f'{len(candidates)} candidates are loaded from {args.candidate_path}')
        topk_preds = [list(filter(lambda x: (x in candidates) or (x.lower() in candidates), a)) for a in predictions]
        topk_preds = [a[:args.top_k] if len(a) > 0 else [''] for a in topk_preds]
        predictions = topk_preds[:]
        top1_preds = [a[0] for a in topk_preds]
        # Track evidence for evaluation
        evidences = [evidences[i][:args.top_k] if len(preds) > 0 else [''] for i, preds in enumerate(topk_preds)]
        top1_evids = [e[0] for e in evidences]
    else:
        predictions = [a[:args.top_k] if len(a) > 0 else [''] for a in predictions]
        top1_preds = [a[0] for a in predictions]
        # Track evidence for evaluation
        evidences = [evidences[i][:args.top_k] if len(preds) > 0 else [''] for i, preds in enumerate(predictions)]
        top1_evids = [e[0] for e in evidences]
    no_ans = sum([a == '' for a in top1_preds])
    logger.info(f'no_ans/all: {no_ans}, {len(top1_preds)}')
    logger.info(f'Evaluating {len(top1_preds)} answers')

    # Get em/f1
    f1s, ems = [], []
    for prediction, groundtruth in zip(top1_preds, answers):
        if len(groundtruth)==0:
            f1s.append(0)
            ems.append(0)
            continue
        f1s.append(max([f1_score(prediction, gt)[0] for gt in groundtruth]))
        ems.append(max([exact_match_score(prediction, gt) for gt in groundtruth]))
    final_f1, final_em = np.mean(f1s), np.mean(ems)
    if not args.regex:
        logger.info('EM: %.2f, F1: %.2f'%(final_em * 100, final_f1 * 100))

    # Top 1/k em (or regex em)
    exact_match_topk = 0
    exact_match_top1 = 0
    f1_score_topk = 0
    f1_score_top1 = 0
    redundant_topk = 0
    pred_out = {}
    agg_pred_out = {}

    # Evidence metrics
    evid_exact_match_topk = 0
    evid_exact_match_top1 = 0
    evid_f1_score_topk = 0
    evid_f1_score_top1 = 0

    # Joint phrase+evidence metrics
    total_phr_substr_evid_f1_topk = 0
    total_phr_substr_evid_f1_top1 = 0

    # Set the phrase metric text for logging
    phr_metric = "em"
    if firsthop:
        phr_metric = "substr"

    for i in range(len(predictions)):
        # For debugging
        if i < 3:
            logger.info(f'{i+1}) {questions[i]}')
            logger.info(f'=> groundtruths: {answers[i]}, top 5 prediction: {predictions[i][:5]}, top 5 evidence: {evidences[i][:5]}')

        match_fn = drqa_regex_match_score if args.regex else \
            (drqa_substr_match_score if firsthop else drqa_exact_match_score)
        em_topk = max([drqa_metric_max_over_ground_truths(
            match_fn, prediction, answers[i]
        ) for prediction in predictions[i][:args.top_k]])
        em_top1 = drqa_metric_max_over_ground_truths(
            match_fn, top1_preds[i], answers[i]
        )
        exact_match_topk += em_topk
        exact_match_top1 += em_top1

        if firsthop:
            # Evaluate evidence (sentence) retrieval
            evid_em_topk = max([drqa_metric_max_over_ground_truths(
                drqa_exact_match_score, evidence, answers[i]
            ) for evidence in evidences[i][:args.top_k]])
            evid_em_top1 = drqa_metric_max_over_ground_truths(
                drqa_exact_match_score, top1_evids[i], answers[i]
            )
            evid_exact_match_topk += evid_em_topk
            evid_exact_match_top1 += evid_em_top1

        # Compute top-k redundancy (could be ill-defined for regex)
        rd_topk = sum([drqa_metric_max_over_ground_truths(
            match_fn, prediction, [predictions[i][0]]
        ) for prediction in predictions[i][:args.top_k]])
        redundant_topk += rd_topk

        f1_topk = 0
        f1_top1 = 0
        if not args.regex:
            match_fn = lambda x, y: f1_score(x, y)[0]
            f1_topk = max([drqa_metric_max_over_ground_truths(
                match_fn, prediction, answers[i]
            ) for prediction in predictions[i][:args.top_k]])
            f1_top1 = drqa_metric_max_over_ground_truths(
                match_fn, top1_preds[i], answers[i]
            )
            f1_score_topk += f1_topk
            f1_score_top1 += f1_top1

            if firsthop:
                # Evaluate evidence (sentence) retrieval
                evid_f1_topk = max([drqa_metric_max_over_ground_truths(
                    match_fn, evidence, answers[i]
                ) for evidence in evidences[i][:args.top_k]])
                evid_f1_top1 = drqa_metric_max_over_ground_truths(
                    match_fn, top1_evids[i], answers[i]
                )
                evid_f1_score_topk += evid_f1_topk
                evid_f1_score_top1 += evid_f1_top1

        if firsthop:
            # Evaluate joint phrase+evidence retrieval
            phr_substr_evid_f1_topk = max([drqa_metric_max_over_ground_truths(
                drqa_substr_f1_match_score, {
                    'pred_substr': predictions[i][k],
                    'pred_f1': evidences[i][k]
                }, answers[i]
            ) for k in range(args.top_k)])
            phr_substr_evid_f1_top1 = drqa_metric_max_over_ground_truths(
                drqa_substr_f1_match_score, {
                    'pred_substr': top1_preds[i],
                    'pred_f1': top1_evids[i]
                }, answers[i]
            )
            total_phr_substr_evid_f1_topk += phr_substr_evid_f1_topk
            total_phr_substr_evid_f1_top1 += phr_substr_evid_f1_top1

        # Score statistics
        assert len(predictions[i]) <= args.top_k
        pred_out[qids[i]] = {
            'question': questions[i],
            'answer': answers[i], 'title': titles[i][0],
            'pred_phrase': predictions[i], 'pred_title': [pt[0] for pt in pred_titles[i]],
            'pred_evidence': evidences[i] if evidences is not None else '',
            'score': scores[i],
            f'{phr_metric}_top1': bool(em_top1), f'{phr_metric}_top{args.top_k}': bool(em_topk),
            'f1_top1': f1_top1, f'f1_top{args.top_k}': f1_topk,
            # 'se_pos': se_positions[i] if se_positions is not None else (-1, -1),
            'rd_topk': rd_topk,
        }
        if firsthop:
            # Add evidence metrics
            evid_pred_out = {
                'evid_em_top1': bool(evid_em_top1),
                f'evid_em_top{args.top_k}': bool(evid_em_topk),
                'evid_f1_top1': evid_f1_top1,
                f'evid_f1_top{args.top_k}': evid_f1_topk
            }
            pred_out[qids[i]].update(evid_pred_out)

            # Add joint metrics
            joint_pred_out = {
                'phr_substr_evid_f1_top1': phr_substr_evid_f1_top1,
                f'phr_substr_evid_f1_top{args.top_k}': phr_substr_evid_f1_topk
            }
            pred_out[qids[i]].update(joint_pred_out)

    # Aggregate prediction metrics
    total = len(predictions)
    exact_match_top1 = 100.0 * exact_match_top1 / total
    f1_score_top1 = 100.0 * f1_score_top1 / total
    logger.info({f'{phr_metric}_top1': exact_match_top1, 'f1_score_top1': f1_score_top1})
    exact_match_topk = 100.0 * exact_match_topk / total
    f1_score_topk = 100.0 * f1_score_topk / total
    logger.info({f'{phr_metric}_top{args.top_k}': exact_match_topk, f'f1_score_top{args.top_k}': f1_score_topk})
    redundant_topk = redundant_topk / total
    logger.info({f'redundancy of top{args.top_k}': redundant_topk})
    if firsthop:
        # Add evidence metrics
        evid_exact_match_top1 = 100.0 * evid_exact_match_top1 / total
        evid_f1_score_top1 = 100.0 * evid_f1_score_top1 / total
        logger.info({'evid_em_top1': evid_exact_match_top1, 'evid_f1_score_top1': evid_f1_score_top1})
        evid_exact_match_topk = 100.0 * evid_exact_match_topk / total
        evid_f1_score_topk = 100.0 * evid_f1_score_topk / total
        logger.info(
            {f'evid_em_top{args.top_k}': evid_exact_match_topk, f'evid_f1_score_top{args.top_k}': evid_f1_score_topk})

        # Add joint metrics
        total_phr_substr_evid_f1_top1 = 100.0 * total_phr_substr_evid_f1_top1 / total
        total_phr_substr_evid_f1_topk = 100.0 * total_phr_substr_evid_f1_topk / total
        logger.info({'phr_substr_evid_f1_top1': total_phr_substr_evid_f1_top1,
                     f'total_phr_substr_evid_f1_top{args.top_k}': total_phr_substr_evid_f1_topk})

    # Store aggregated metrics in a separate file
    agg_pred_out = {
        'total': total,
        f'{phr_metric}_top1': exact_match_top1,
        'f1_score_top1': f1_score_top1,
        f'{phr_metric}_top{args.top_k}': exact_match_topk,
        f'f1_score_top{args.top_k}': f1_score_topk,
        f'redundancy of top{args.top_k}': redundant_topk,
    }
    if firsthop:
        evid_agg_pred_out = {
            'evid_em_top1': evid_exact_match_top1,
            'evid_f1_score_top1': evid_f1_score_top1,
            f'evid_em_top{args.top_k}': evid_exact_match_topk,
            f'evid_f1_score_top{args.top_k}': evid_f1_score_topk
        }
        agg_pred_out.update(evid_agg_pred_out)
        joint_agg_pred_out = {
            'phr_substr_evid_f1_top1': total_phr_substr_evid_f1_top1,
            f'total_phr_substr_evid_f1_top{args.top_k}': total_phr_substr_evid_f1_topk
        }
        agg_pred_out.update(joint_agg_pred_out)

    # Dump predictions
    if save_path is not None:
        pred_dir = os.path.join(save_path, 'pred')
    else:
        if len(args.load_dir) == 0:
            pred_dir = os.path.join(os.environ['SAVE_DIR'], 'pred')
        else:
            pred_dir = os.path.join(args.load_dir, 'pred')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    save_pred = save_pred if type(save_pred) is bool else args.save_pred
    if save_pred:
        pred_path = os.path.join(
            pred_dir,
            os.path.splitext(os.path.basename(data_path))[0] +
            f'_{total}_top{args.top_k}' +
            ('' if pred_fname_suffix == '' else f'_{pred_fname_suffix}') +
            '.pred'
        )
        agg_pred_path = pred_path.replace('.pred', '_agg.pred')
        logger.info(f'Saving individual predictions to {pred_path}')
        with open(pred_path, 'w') as f:
            json.dump(pred_out, f, indent=2)
        logger.info(f'Saving aggregate predictions to {agg_pred_path}')
        with open(agg_pred_path, 'w') as f:
            json.dump(agg_pred_out, f, indent=2)

    # Evaluate passage retrieval
    if args.eval_psg:
        evaluate_results_psg(pred_path, args)

    return_tuple = (exact_match_top1, f1_score_top1, exact_match_topk, f1_score_topk)
    if firsthop:
        evid_return_tuple = (evid_exact_match_top1, evid_f1_score_top1, evid_exact_match_topk, evid_f1_score_topk)
        joint_return_tuple = (total_phr_substr_evid_f1_top1, total_phr_substr_evid_f1_topk)
        return return_tuple, evid_return_tuple, joint_return_tuple
    return return_tuple


def evaluate_results_kilt(predictions, qids, questions, answers, titles, args, evidences,
                          scores, pred_titles, se_positions=None, save_pred=False, pred_fname_suffix="",
                          data_path=None, save_path=None):
    total=len(predictions)

    # load title2id dict and convert predicted titles into wikipedia_ids
    with open(args.title2wikiid_path) as f:
        title2wikiid = json.load(f)
    pred_wikipedia_ids = [[[title2wikiid[t] for t in title_] for title_ in title] for title in pred_titles]

    # dump official predictions
    if len(args.load_dir) == 0:
        pred_dir = os.path.join(os.environ['SAVE_DIR'], 'pred-kilt')
    else:
        pred_dir = os.path.join(args.load_dir, 'pred-kilt')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    pred_official_path = os.path.join(
        pred_dir, f'{args.load_dir.split("/")[-1]}_' +
        os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}.jsonl'
    )
    official_preds_to_save = []
    for prediction, title, question, pred_wikipedia_id, qid in zip(predictions, pred_titles, questions, pred_wikipedia_ids, qids):
        if ("wned" in pred_official_path or
            "cweb" in pred_official_path or
            "aidayago2" in pred_official_path):
            answer = title[0][0]
        else:
            answer = prediction[0].strip(string.punctuation)

        output = {
            'answer': answer,
            'provenance': [{'wikipedia_id': pred_wid_} for pred_wid in pred_wikipedia_id for pred_wid_ in pred_wid]
        }
        official_preds_to_save.append({
            'id': qid,
            'input': question,
            'output': [output]
        })

    logger.info(f'Saving official prediction file to {pred_official_path}')
    kilt_store_data(pred_official_path, official_preds_to_save)

    assert '.jsonl' in args.kilt_gold_path, "kilt_gold_path should be .jsonl"
    result = kilt_evaluate(
        gold=args.kilt_gold_path,
        guess=pred_official_path)

    # logging results
    result_to_logging = {
        'accuracy':result['downstream']['accuracy'],
        'f1':result['downstream']['f1'],
        'KILT-accuracy':result['kilt']['KILT-accuracy'],
        'KILT-f1':result['kilt']['KILT-f1'],
        'Rprec':result['retrieval']['Rprec'],
        'recall@5':result['retrieval']['recall@5']
    }

    logger.info(result_to_logging)

    # make custom predictions
    pred_out = {}
    for i in range(len(predictions)):
        # For debugging
        if i < 3:
            logger.info(f'{i+1}) {questions[i]}')
            logger.info(f'=> groundtruths: {answers[i]}, top 5 prediction: {predictions[i][:5]}')

        guess_answer = predictions[i][0]
        gold_candidate_answers = answers[i]
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1
        
        pred_out[qids[i]] = {
                'question': questions[i],
                'answer': answers[i], 'prediction': predictions[i], 'score': scores[i], 'title': pred_titles[i],
                'evidence': evidences[i] if evidences is not None else '',
                'em_top1': bool(local_accuracy),
        }

    # dump custom predictions
    pred_path = os.path.join(
        pred_dir, os.path.splitext(os.path.basename(args.test_path))[0] + f'_{total}.pred'
    )
    logger.info(f'Saving custom prediction file to {pred_path}')
    with open(pred_path, 'w') as f:
        json.dump(pred_out, f)

    return result['retrieval']['Rprec'], result['retrieval']['recall@5'], result['kilt']['KILT-accuracy'], result['kilt']['KILT-f1']


def evaluate_results_psg(pred_path, args):
    # Read prediction
    my_pred = json.load(open(pred_path))

    my_target = []
    avg_len = []
    for qid, pred in tqdm(enumerate(my_pred.values())):
        my_dict = {"id": str(qid), "question": None, "answers": [], "ctxs": []}

        # truncate
        pred = {key: val[:args.psg_top_k] if key in ['evidence', 'title', 'se_pos', 'prediction'] else val for key, val in pred.items()}

        # TODO: need to add id for predictions.pred
        my_dict["question"] = pred["question"]
        my_dict["answers"] = pred["answer"]
        pred["title"] = [titles[0] for titles in pred["title"]]

        assert len(set(pred["evidence"])) == len(pred["evidence"]) == len(pred["title"]), "Should use opt2 for aggregation"
        # assert all(pr in evd for pr, evd in zip(pred["prediction"], pred["evidence"]))  # prediction included TODO: fails when return_sent=True

        # Pad up to top-k
        if not(len(pred["prediction"]) == len(pred["evidence"]) == len(pred["title"]) == args.psg_top_k):
            assert len(pred["prediction"]) == len(pred["evidence"]) == len(pred["title"]) < args.psg_top_k, \
                (len(pred["prediction"]), len(pred["evidence"]), len(pred["title"]))
            # logger.info(len(pred["prediction"]), len(pred["evidence"]), len(pred["title"]))

            pred["evidence"] += [pred["evidence"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            pred["title"] += [pred["title"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            pred["se_pos"] += [pred["se_pos"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            pred["prediction"] += [pred["prediction"][-1]] * (args.psg_top_k - len(pred["prediction"]))
            assert len(pred["prediction"]) == len(pred["evidence"]) == len(pred["title"]) == args.psg_top_k

        # Used for markers
        START = '<p_start>'
        END = '<p_end>'
        se_idxs = [[se_pos[0], max(se_pos[0], se_pos[1])] for se_pos in pred["se_pos"]]

        # cut based on max psg len
        my_dict["ctxs"] = [
            {"title": title, "text": ' '.join(evd.split()[:args.max_psg_len])}
            for evd, title in zip(pred["evidence"], pred["title"])
        ]

        # Add markers for predicted phrases
        if args.mark_phrase:
            my_dict["ctxs"] = [
                {"title": ctx["title"], "text": ctx["text"][:se[0]] + f"{START} " + ctx["text"][se[0]:se[1]] + f" {END}" + ctx["text"][se[1]:]}
                for ctx, se in zip(my_dict["ctxs"], se_idxs)
            ]

        my_target.append(my_dict)
        avg_len += [len(ctx['text'].split()) for ctx in my_dict["ctxs"]]
        assert len(my_dict["ctxs"]) == args.psg_top_k
        assert all(len(ctx['text'].split()) <= args.max_psg_len for ctx in my_dict["ctxs"])

    logger.info(f"avg psg len={sum(avg_len)/len(avg_len):.2f} for {len(my_pred)} preds")

    out_file = os.path.join(
        os.environ['SAVE_DIR'], os.path.basename(args.load_dir), 'pred',
        os.path.splitext(os.path.basename(pred_path))[0] + 
        f'_{"sent" if args.return_sent else "psg"}-top{args.psg_top_k}{"_mark" if args.mark_phrase else ""}.json'
    )
    logger.info(f"dump to {out_file}")
    json.dump(my_target, open(out_file, 'w'), indent=4)

    # Call subprocess for evaluation
    command = f'python scripts/postprocess/recall.py --k_values 1,5,20,100 --results_file {out_file} --ans_fn string'
    subprocess.run(command.split(' '))


if __name__ == '__main__':
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_index_options()
    options.add_retrieval_options()
    options.add_data_options()
    args = options.parse()

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_mode == 'eval':
        evaluate(args)

    elif args.run_mode == 'eval_all':
        # Load MIPS & query encoder
        mips = load_phrase_index(args)
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer, _ = load_encoder(device, args)

        # Evaluate all test sets
        test_paths = args.test_path.split(',')
        assert all(os.path.exists(path) for path in test_paths)
        logger.info(f"Evaluating {len(test_paths)} datasets: {test_paths}")
        ems = []
        for test_path in test_paths:
            logger.info(f"Evaluating {test_path}")
            new_args = copy.deepcopy(args)
            new_args.test_path = test_path
            if 'trec' in test_path:
                new_args.regex = True
                logger.info('Enable regex for TREC')
            if 'webq' in test_path:
                new_args.candidate_path = os.path.join(os.environ['DATA_DIR'], 'open-qa/webq/freebase-entities.txt')
                logger.info('Enable candidates for WebQuestions')
            em, _, _, _ = evaluate(new_args, mips, query_encoder, tokenizer)
            ems.append(f'{em:.1f}')
        logger.info(f"Results of {args.load_dir}")
        logger.info(f'Top1 EMs: {" ".join(ems)}')
    
    else:
        raise NotImplementedError
