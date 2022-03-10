from IPython import embed
from tqdm import tqdm
import json
import numpy as np

from densephrases import DensePhrases


# Arguments: Presently hard-coded; TODO: Add argparse later on
top_k = 2
n_sel = 2
batch_size = 100
load_dir = 'princeton-nlp/densephrases-multi-query-multi'
dump_dir = '/home/nishantraj_umass_edu/DAQuA-Difficulty-Aware-Question-Answering/DPhrases/outputs/densephrases-multi_wiki-20181220/dump'
idx_name = 'start/1048576_flat_OPQ96_small'
data_path = "/gypsum/scratch1/dagarwal/multihop_dense_retrieval/data/hotpot/hotpot_qas_val.json"
device = 'cuda'  # 'cpu'
ret_unit='phrase'
out_file = f'predictions_{__import__("calendar").timegm(gmt)}.json'
# Questions to be updated based on a function to read json validation files and get list of queries after reading
questions = ["What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",\
"Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?"]


def load_densephrase_module(load_dir=load_dir, dump_dir=dump_dir, index_name=idx_name, device=device):
    """
    Load Dense Phrases module
    """
    model = DensePhrases(load_dir=load_dir, dump_dir=dump_dir, index_name=idx_name, device=device)
    return model


def get_first_hop_phrase_score_list(metadata, top_k):
    """
    Get list of scores retrieved from first hop. Note if there are [q1, q2] queries and
    we requested 2 phrases (top_k) corresponding to these i.e. [[s11,s12],[s21,s22]]
    """
    phrase_score_list = []
    for i in range(len(metadata)):
        interim_scores = []
        for j in range(top_k):
            interim_scores.append(metadata[i][j]['score'])
        phrase_score_list.append(interim_scores)
    return phrase_score_list


def create_new_query(query_list, top_k, phrases):
    """
    Add query to the first hop retrievals and treat them as queries for second hop retrievals
    """
    all_query_pairs = []
    for query_num in range(len(query_list)):
        interim_comb = []
        for sub_phrase_num in range(top_k):
            interim_comb.append(phrases[query_num][sub_phrase_num] + " " + query_list[query_num])
        all_query_pairs.append(interim_comb)

    # Returns a flattened query list for second hop retrieval
    flat_second_hop_qlist = [query for blist in all_query_pairs for query in blist]

    return flat_second_hop_qlist


def get_second_hop_retrieval(hop_phrases, hop_metadata, n_ins, top_k):
    """
    Returns scores and phrases retrieved as a result of retrieval from second hop.
    Required for deciding best chain for the reader module
    """
    hop_phrases_flat = [phrase_val for li in hop_phrases for phrase_val in li]
    hop_phrase_score_flat = [sub['score'] for mdata in hop_metadata for sub in mdata]
    phrase_sc_2 = np.array(hop_phrase_score_flat).reshape((n_ins, top_k, top_k))
    phrase_arr_2 = np.array(hop_phrases_flat).reshape((n_ins, top_k, top_k))

    return phrase_sc_2, phrase_arr_2


def get_top_chains(scores_1, scores_2, doc_id_1, doc_id_2, top_k, n_sel):
    """
    Get the scores for a single question at hop 1 and hop 2 and for corresponding phrase combination,
    return the combination of phrase at hop 1 and hop 2 that gave best overall score product.
    """
    path_scores = np.expand_dims(scores_1, axis=2) * scores_2
    search_scores = np.squeeze(path_scores)
    ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1], 
                                                (top_k, top_k))).transpose()
    chains, answers = [], []
    for _ in range(n_sel):
        path_ids = ranked_pairs[_]
        did_1 = doc_id_1[0][path_ids[0]]
        did_2 = doc_id_2[0][path_ids[0]][path_ids[1]]
        chains.append([did_1, did_2])
        answers.append(did_2)

    return chains, answers

def run_chain_all_queries(query_list, phrase_sc_1, phrase_sc_2, phrase_arr_1, phrase_arr_2, top_k, n_sel, answers=None):
    """
    For a given query, find the # of n_sel chains based on top_k phrases from Dense Phrases module
    """
    qchain_arr = []
    for num_q in range(len(query_list)):
        # Get scores and phrase combinations here
        scores_1 = np.array(phrase_sc_1[num_q]).reshape(1, top_k)
        scores_2 = np.array(phrase_sc_2[num_q]).reshape(1, top_k, top_k)
        doc_id_1 = np.array(phrase_arr_1[num_q]).reshape(1, top_k)
        doc_id_2 = np.array(phrase_arr_2[num_q]).reshape(1, top_k, top_k)

        # Use Get Top Chain Function to retrieve best chains for this question
        chain, preds = get_top_chains(scores_1, scores_2, doc_id_1, doc_id_2, top_k, n_sel)

        obj = {
            "query": query_list[num_q],
            "gold_answer": answers[num_q] if answers is not None else None,
            "predicted_answers": preds,
            "predicted_chains": chain,
        }
        qchain_arr.append(obj)
    return qchain_arr


def prune_to_topk(obj, topk):
    res = []
    for o in obj:
        res.append(o[:topk])
    return res


def get_key(dictionary, key, size=-1):
    if size < 0:
        return [d[key] for d in dictionary]
    res = []
    for i, d in enumerate(dictionary):
        if i >= size:
            return res
        res.append(d[key])
    return res


def read_queries(path):
    res = []
    with open(path, 'r') as fp:
        print(f"Loading data from {path}")
        lines = fp.readlines()
        for l in lines:
            obj = json.loads(l)
            if obj["type"] == "bridge":
                res.append(obj)
    print(f"Found {len(res)} bridge questions in {len(lines)} total questions.")
    return res


def run_batch_inference(model, questions=None, answers=None, write=False, silent=True):
    # Total Number of Queries
    n_ins = len(questions)

    # Run the model to retrieve the first hop of phrases and their corresponding scores
    if not silent:
        print("Running DensePhrases module for first hop phrase retrieval ...")
    phrases, metadata = model.search(questions, retrieval_unit=ret_unit, top_k=top_k+1, return_meta=True)
    phrases = prune_to_topk(phrases, top_k)
    metadata = prune_to_topk(metadata, top_k)


    # Get phrase scores separately from metadata for first hop
    phrase_sc_1 = get_first_hop_phrase_score_list(metadata, top_k)

    # Get new query combinations for second hop retrieval from DensePhrases module
    flat_second_hop_qlist = create_new_query(questions, top_k, phrases)

    # Run second hop retrieval from DensePhrases module
    if not silent:
        print("Running second hop of phrase retrieval")
    hop_phrases, hop_metadata = model.search(flat_second_hop_qlist, retrieval_unit=ret_unit, top_k=top_k+1,
                                             return_meta=True)
    hop_phrases = prune_to_topk(hop_phrases, top_k)
    hop_metadata = prune_to_topk(hop_metadata, top_k)


    # Get score and phrase list from second retrieval for evidence chain extraction
    phrase_sc_2, phrase_arr_2 = get_second_hop_retrieval(hop_phrases, hop_metadata, n_ins, top_k)

    if not silent:
        print("Creating final dictionary")
    # Get final chain of best n_sel queries for each question
    qchain_arr = run_chain_all_queries(questions, phrase_sc_1, phrase_sc_2, 
                                       phrases, phrase_arr_2, top_k, n_sel,
                                       answers)

    if write:
        # Dump information inside a JSON file
        with open(out_file, 'w') as fp:
            json.dump(qchain_arr, fp, indent=4)

    return qchain_arr


if __name__ == "__main__":

    # Load the DensePhrases module
    print("Loading DensePhrases module...")
    model = load_densephrase_module(load_dir=load_dir, dump_dir=dump_dir, index_name=idx_name, device=device)

    # Read questions and answers from data_path
    queries = read_queries(data_path)
    questions = get_key(queries, 'question')
    answers = get_key(queries, 'answer')
    answers = [answer[0] for answer in answers]  # flattening

    # Run batched inference
    results = []
    print("Running batched multi-hop inference...")
    for i in tqdm(range(0, len(questions), batch_size)):
        batch_results = run_batch_inference(model, questions=questions[i:i+batch_size], answers=answers[i:i+batch_size], write=False)
        results += batch_results

    # Write predictions to disk
    with open(out_file, 'w') as fp:
        json.dump(results, fp, indent=4)
    print(f"Predictions saved at {out_file}")
