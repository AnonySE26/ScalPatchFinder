from pandas import read_csv
import re
import tqdm
from functools import partial
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import voyageai
import os
import sys
import multiprocessing
import argparse

def get_detailed_instruct(query: str) -> str:
    INSTRUCTION = "Given the description of a security vulnerability, retrieve the patching commit (commit message + diff code)."
    return f'Instruct: {INSTRUCTION}\nQuery: {query}'

def return_highlighted_desc(sentences_batch, cve, cve2highlight):
    highlight = cve2highlight.get(cve, "None")
    if type(highlight) == float:
        highlight = "None"
    if highlight != "None":
        start_index = sentences_batch.find(highlight)
        if start_index == -1:
            return -1, -1
        end_index = start_index + len(highlight) - 1
        return start_index, end_index
    return -1, -1

def is_overlap(range1, range2):
   if range1 == (-1, -1): return False
   if range2 == (-1, -1): return False
   if min(range1) > max(range2) or min(range2) > max(range1):
       return False
   return True

def get_embedding_query(each_cve, desc_text, model_name, INSTRUCTION, model=None, tokenizer=None, context_window=512, cve2highlight=None, eta=0.9):
    if model_name == "voyage":
        query_embedding = vo.embed([desc_text], model = "voyage-3", input_type="document").embeddings[0]
    elif model_name.startswith("grit"):
        this_instruction = gritlm_instruction(INSTRUCTION)
        sentences_batch = this_instruction + desc_text + model.embed_eos
        sentences_batch = re.sub("\s+", " ", sentences_batch)
        tokens = model.tokenizer([sentences_batch],
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length= context_window, return_offsets_mapping=True).to(model.device)
        start_index, end_index = return_highlighted_desc(sentences_batch, each_cve, cve2highlight)
        char_positions = tokens["offset_mapping"][0][1:]
        is_highlight = []
        for (each_start, each_end) in char_positions:
            if is_overlap((each_start, each_end), (start_index, end_index)):
                is_highlight.append(1)
            else:
                is_highlight.append(0)
        token_str = model.tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])[1:]
        #raise Exception(cve2highlight.get(each_cve), [token_str[x] for x in range(len(token_str)) if is_highlight[x] == 1], is_highlight, start_index, end_index, char_positions)
        assert len(char_positions) == len(is_highlight) == len(tokens["input_ids"][0]) - 1
        # is_highlight = [0] + is_highlight
        # is_highlight_sum = sum(is_highlight)
        # if is_highlight_sum > 0:
        #     attention_weight = [eta + (1-eta) * is_highlight[x] if tokens["attention_mask"][0][x] == 1 else 0 for x in range(len(is_highlight))]
        #     coeff = float(torch.sum(tokens["attention_mask"][0])) / sum(attention_weight)
        #     attention_weights = [coeff * x for x in attention_weight]
        # else:
        attention_weights = [1] * len(tokens["input_ids"][0])

        query_embedding = model.encode([desc_text], instruction=this_instruction, batch_size = 1, max_length = context_window, this_length = len(tokens["input_ids"][0]), attention_weights = attention_weights)[0] 
    elif model_name == "salesforce":
        query_embedding = get_embedding(model, tokenizer, [get_detailed_instruct(desc_text)], 1)[0] 
    elif model_name == "modernbert":
        query_embedding = model.encode([desc_text], batch_size=1)[0]
    return query_embedding

INSTRUCTION = "Represent this CVE description to retrieve the commit (commit message + diff code) that patches this CVE. "
#INSTRUCTION = "Given the description of a security vulnerability, retrieve the patching commit (commit message + diff code). " #f"Given the description of the following security vulnerability of {owner}/{repo}, retrieve the patching commit. Here is the vulnerability description: "


def process_cve_group(owner_repo, cve2desc, model_name, model, tokenizer, data, context_window, cve2highlight, eta):
    owner, repo = owner_repo[0], owner_repo[1]
    cves = set(data[(data.owner == owner) & (data.repo == repo)]["cve"])
    cve2embedding = {}

    for each_cve in cves:
        if each_cve not in cve2desc:
            print(f"Skipping {each_cve} as it does not exist in cve2desc")
            continue
        desc_text = cve2desc[each_cve]
        embedding = get_embedding_query(each_cve, desc_text, model_name, INSTRUCTION, model, tokenizer, context_window, cve2highlight, eta)
        cve2embedding[each_cve] = embedding if isinstance(embedding, list) else embedding.tolist()

    output_path = f"../../../feature/{owner}@@{repo}/{model_name}/cve2embedding.json"
    with open(output_path, "w") as f:
        json.dump(cve2embedding, f)

    return (owner, repo, len(cve2embedding)) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="grit_instruct_512_file")
    parser.add_argument("--dataset_name", type=str, default="AD")
    parser.add_argument("--is_train", type=bool, action="store_false")
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    context_window = 512 #int(sys.argv[3])
    fold = "train" if args.is_train else "test"
    #fold = "train" if is_train else "test"

    print("fold", fold, flush=True)

    eta = 0.6

    repo2cve2negcommits = json.load(open(f"../../../feature/repo2cve2negcommits_{dataset_name}_500_unsampled.json" if dataset_name == "patchfinder" else f"../../../feature/repo2cve2negcommits_{dataset_name}_500.json", "r"))

    if model_name != "voyage":
        from gritlm import GritLM
        import gritlm
        print(gritlm.__file__)
        from index_commits import gritlm_instruction, get_embedding
        import torch
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer, util

    if model_name.startswith("grit"):
        model = GritLM("GritLM/GritLM-7B", torch_dtype="auto") 
        tokenizer = None
    else:
        model = None
        tokenizer = None

    voyageai.api_key = json.load(open("../../secret.json", "r"))["voyage"]
    vo = voyageai.Client()
    
    cve2desc = json.load(open("../../../NVD/cve2desc.json", "r"))

    if dataset_name == "AD":
        highlight_data = read_csv(f"../csv/highlighted_{dataset_name}_test.csv")
        cve2highlight = {} # {highlight_data.iloc[x]["CVE"]: highlight_data.iloc[x]["manual"] for x in range(len(highlight_data))}
    else:
        cve2highlight = {}
    
    cve2commits = {}
    data = read_csv(f"../csv/{dataset_name}_{fold}.csv")
    groupby_list = list(data.groupby(["owner", "repo"]))

    if model_name == "voyage":
        with multiprocessing.Pool(processes=10) as pool:
            process_func = partial(process_cve_group, data=data, cve2desc=cve2desc, model_name=model_name, model=model, tokenizer=tokenizer, context_window = context_window, cve2highlight=cve2highlight, eta=eta)
            results = list(tqdm.tqdm(pool.imap_unordered(process_func, [(owner, repo) for (owner, repo), _ in groupby_list]), total=len(groupby_list)))
    else:
        for owner_repo, _ in tqdm.tqdm(groupby_list):
            if not os.path.exists(f"../../../feature/{owner_repo[0]}@@{owner_repo[1]}/{model_name}/"):
                continue
            if os.path.exists(f"../../../feature/{owner_repo[0]}@@{owner_repo[1]}/{model_name}/cve2embedding.json"):
                continue
            if args.is_train and (owner_repo[0] + "@@" + owner_repo[1] not in repo2cve2negcommits): continue
            #if owner_repo != ("apache", "tomcat"): continue
            process_cve_group(owner_repo, cve2desc, model_name, model, tokenizer, data, context_window, cve2highlight, eta)

