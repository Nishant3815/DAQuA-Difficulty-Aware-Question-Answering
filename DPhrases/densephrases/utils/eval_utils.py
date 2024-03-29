import sys
import ujson as json
import re
import string
import unicodedata
import pickle
from collections import Counter

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def drqa_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def drqa_exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def drqa_substr_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) substring match with the ground truth."""
    norm_pred = normalize_answer(prediction)
    return norm_pred != '' and norm_pred in normalize_answer(ground_truth)


def drqa_f1_match_score(prediction, ground_truth, f1_threshold=None):
    """Check if the f1 score of the prediction and the ground truth is above a given threshold."""
    if f1_threshold is None:
        # Return the f1 score if there is a substr match; else return 0
        return f1_score(prediction, ground_truth)[0]
    return f1_score(prediction, ground_truth)[0] >= f1_threshold


def drqa_substr_exact_match_score(pred_substr, pred_exact, ground_truth):
    """
    Check if pred_substr is a (soft) substring match with the ground truth
    AND
    if pred_exact is a (soft) exact match with the ground truth
    """
    return drqa_substr_match_score(pred_substr, ground_truth) and drqa_exact_match_score(pred_exact, ground_truth)


def drqa_substr_f1_match_score(pred_substr, pred_f1, ground_truth, f1_threshold=None):
    """
    Check if pred_substr is a (soft) substring match with the ground truth
    AND
    if the token (soft) f1 score of the pred_f1 with the ground truth is at least as high as the threshold
    """
    if f1_threshold is None:
        # Return the f1 score if there is a substr match; else return 0
        return drqa_substr_match_score(pred_substr, ground_truth) * f1_score(pred_f1, ground_truth)[0]
    return drqa_substr_match_score(pred_substr, ground_truth) and (f1_score(pred_f1, ground_truth)[0] >= f1_threshold)


def drqa_regex_match_score(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException as e:
        # logger.warn('Regular expression failed to compile: %s' % pattern)
        # print('re failed to compile: [%s] due to [%s]' % (pattern, e))
        return False
    return compiled.match(prediction) is not None


def drqa_metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if not type(prediction) is dict:
            score = metric_fn(prediction, ground_truth)
        else:
            score = metric_fn(**prediction, ground_truth=ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += em
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall


def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for dp in gold:
        cur_id = dp['_id']
        em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)


def analyze(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for dp in gold:
        cur_id = dp['_id']

        em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if (prec + recall == 0):
            f1 = 0
        else:
            f1 = 2 * prec * recall / (prec+recall)

        print (dp['answer'], prediction['answer'][cur_id])
        print (f1, em)
        a = input()


if __name__ == '__main__':
    #eval(sys.argv[1], sys.argv[2])
    analyze(sys.argv[1], sys.argv[2])
