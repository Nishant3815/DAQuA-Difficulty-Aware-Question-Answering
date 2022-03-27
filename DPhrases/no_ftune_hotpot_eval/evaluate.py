from IPython import embed
import json
import unicodedata

# Arguments: Presently hard-coded; TODO: Add argparse later on
PREDICTIONS_PATH = "./data.json"


def read_results(path):
    with open(path, 'r') as jsonfile:
        results = json.load(jsonfile)
    return results


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def lower_list(l):
    return list(map(lambda x: str.lower(normalize(x)), l))


def metric_em(gold, preds):
    """
    Hit when gold answer exactly matches any of the predictions
    """
    if gold in preds:
        return 1
    return 0


def metric_substr_gp(gold, preds):
    """
    Hit when gold answer is a substring of any of the predictions
    """
    for p in preds:
        if gold in p:
            return 1
    return 0


def metric_substr_pg(gold, preds):
    """
    Hit when any of the predictions is a substring of the gold answer
    """
    for p in preds:
        if p in gold:
            return 1
    return 0


def metric_substr_2way(gold, preds):
    """
    Hit when either the gold answer or any of the predictions is a substring of each other
    """
    for p in preds:
        if p in gold or gold in p:
            return 1
    return 0


def overlap_coeff(gold, preds):
    """
    Overlap (Szymkiewicz-Simpson) coefficient
    Ref: https://aircconline.com/mlaij/V3N1/3116mlaij03.pdf
    """
    best = 0.
    gold_tokens = set(gold.split())
    for p in preds:
        p_tokens = set(p.split())
        try:
            val = len(gold_tokens.intersection(p_tokens)) / min(len(gold_tokens), len(p_tokens))
        except:
            val = 0
        best = max(best, val)
    return best


def write_eval_results(path, obj):
    with open(path, 'w') as fp:
        json.dump(obj, fp, indent=4)

        
def compute_metrics(results):
    """
    Run each evaluation metric on the predictions and aggregate results
    """
    total = len(results)
    metrics = {
        "em": metric_em, 
        "substr_gp": metric_substr_gp, 
        "substr_pg": metric_substr_pg, 
        "substr2": metric_substr_2way, 
        "overlap": overlap_coeff
    }
    eval_dict = {
        "n_questions": total,
    }
    for m in metrics:
        eval_dict[m] = 0.

    # Run evaluations
    for r in results:
        gold = normalize(r['gold_answer']).lower()
        preds = lower_list(r['predicted_answers'])
        # Accumulate
        for m in metrics:
            eval_dict[m] += metrics[m](gold, preds)
    # Normalize
    overall = 0.
    for m in metrics:
        eval_dict[m] = round(eval_dict[m]/total * 100, 2)
        overall += eval_dict[m]
    overall /= len(metrics)
    eval_dict['overall_avg'] = round(overall, 2)

    print(json.dumps(eval_dict, indent=2))
    return eval_dict


def compare_predictions(first, second, unique=True, union_gold=False):
    total = len(first)
    metrics = {
        "em": metric_em, 
        "substr_gp": metric_substr_gp, 
        "substr_pg": metric_substr_pg, 
        "substr2": metric_substr_2way, 
        "overlap": overlap_coeff
    }
    eval_dict = {
        "n_questions": total,
    }
    for m in metrics:
        eval_dict[m] = 0.
        
    norm_const = 0
    for i in range(total):
        first_preds = list(set(first[i]['predicted_answers'])) if unique else first[i]['predicted_answers']
        second_preds = list(set(second[i]['predicted_answers'])) if unique else second[i]['predicted_answers']
        if not union_gold:
            for f in first_preds:
                norm_const += 1
                gold = normalize(f).lower()
                preds = lower_list(second_preds)
                # Accumulate
                for m in metrics:
                    eval_dict[m] += metrics[m](gold, preds)
        else:
            norm_const += 1
            union = list(set(first_preds + second_preds))
            gold = normalize(first[i]['gold_answer']).lower()
            preds = lower_list(union)
            # Accumulate
            for m in metrics:
                eval_dict[m] += metrics[m](gold, preds)
            
    # Normalize
    overall = 0.
    for m in metrics:
        eval_dict[m] = round(eval_dict[m] / norm_const * 100, 2)
        overall += eval_dict[m]
    overall /= len(metrics)
    eval_dict['overall_avg'] = overall

    print(json.dumps(eval_dict, indent=2))
    return eval_dict
        
    
if __name__ == "__main__":
    # Read data from disk
    print(f"Reading predictions from {PREDICTIONS_PATH}")
    results = read_results(PREDICTIONS_PATH)
    OUTPUT_PATH = PREDICTIONS_PATH.replace(".json", "_eval.json")

    # Compute metrics
    eval_results = compute_metrics(results)

    # Write results to disk
    write_eval_results(OUTPUT_PATH, eval_results)
    print(f"\nSaved results at {OUTPUT_PATH}")
