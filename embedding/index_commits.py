#!/bin/python
import re
#import tiktoken
import tqdm
import os
import json
import subprocess
import voyageai
import sys
import argparse
import multiprocessing
from pandas import read_csv
from utils import get_commit_list_train, get_commit_list_test

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#openai_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

linecount = 0
import time

t1 = time.time()

voyage_max_tokens = 300000
grit_batch_size = 10 #int(sys.argv[4])
#context_window = 256 #int(sys.argv[4])
feature_path = "../feature/" # if len(sys.argv[1]) > 0 else ""
input_data_path = "../repo2commits_diff/" # if len(sys.argv[1]) > 0 else ""
INSTRUCTION = "This is a commit (commit message + diff code) of a repository. Represent it to retrieve the patching commit for a CVE description. "

 
def truncate_to_max_tokens(documents, commits, max_tokens=5000):
    truncated_documents = []
    truncated_commits = []
    total_tokens = 0

    for doc, commit in zip(documents, commits):
        doc_tokens = doc.split()
        if total_tokens + len(doc_tokens) <= max_tokens:
            truncated_documents.append(doc)
            truncated_commits.append(commit)
            total_tokens += len(doc_tokens)
        else:
            remaining_tokens = max_tokens - total_tokens
            truncated_documents.append(" ".join(doc_tokens[:remaining_tokens]))
            truncated_commits.append(commit)
            break  
    return truncated_documents, truncated_commits

def gritlm_instruction(INSTRUCTION):
    #instruction = "Given the description of a security vulnerability, retrieve the patching commit (commit message + diff code)."
    return "<|user|>\n" + INSTRUCTION + "\n<|embed|>\n" if INSTRUCTION else "<|embed|>\n"

def process_task_huggingface(task, model, batch_id, fout, model_name, context_window):

    documents = [x[1] for x in task]
    commits = [x[0] for x in task]
    commit2embedding = {}

    try:
        # No need to add instruction for retrieval documents
        t1 = time.time()
        documents_embeddings = model.encode(documents, instruction=gritlm_instruction(INSTRUCTION), batch_size = 2, max_length = context_window)
        for x in range(len(commits)):
            commit2embedding[commits[x]] = documents_embeddings[x].tolist()
        print("place F", flush=True)
        return commit2embedding
    except Exception as e:
        import traceback
        error_msg = f"Batch {batch_id} Error: {str(e)}\nTraceback:\n{traceback.format_exc()}\n"
        
        print(error_msg, file=sys.stderr)
        sys.stderr.flush()
        if fout:
            fout.write(error_msg)
            fout.flush()
        return {}

def process_task_voyage(task):

    documents = [x[1] for x in task]
    commits = [x[0] for x in task]
    idx_list = [str(x[2]) for x in task]

    vo = voyageai.Client()

    commit2embedding = {}
    try:
        documents, commits = truncate_to_max_tokens(documents,commits, voyage_max_tokens)
        
        documents_embeddings = vo.embed(documents, model = "voyage-3", input_type="document").embeddings
        for x in range(len(commits)):
            commit2embedding[commits[x]] = list(documents_embeddings[x])
        return commit2embedding
    except Exception as e:
        print("error in",  str(e), flush=True)
        return {}

def process_commits_huggingface(chunk, commit2codemsg, commit_list, shared_task_l, task_l_lock, shared_total_token_count):
    local_task_l = []
    current_batch = []
    for each_commit_idx in tqdm.tqdm(chunk):
        each_commit = commit_list[each_commit_idx]
        this_entry = commit2codemsg.get(each_commit, {})
        diff = this_entry.get("diff", "")
        msg = this_entry.get("commit_msg", "")

        this_str = f"Commit message: {msg}\nDiff code: {diff}"
        this_str = re.sub(r"\s+", " ", this_str.replace("\n", " "))[:50000]

        if len(current_batch) < grit_batch_size:
            current_batch.append((each_commit, this_str, each_commit_idx))
        else:
            local_task_l.append(current_batch)
            current_batch = [(each_commit, this_str, each_commit_idx)]
    if len(current_batch) > 0:
        local_task_l.append(current_batch)

    with task_l_lock:
        shared_task_l.extend(local_task_l)

def process_commits_voyage(chunk, commit2codemsg, commit_list, shared_task_l, task_l_lock, shared_total_token_count):
    """ Process a chunk of commits while updating shared memory """
    local_task_l = []  # Local task list for this process
    current_batch = []
    current_batch_tokens = []
    local_total_token_count = 0

    home_dir = "../../"
    voyageai.api_key = json.load(open(home_dir + "/secret.json", "r"))["voyage"]

    vo = voyageai.Client()

    for each_commit_idx in tqdm.tqdm(chunk):
        each_commit = commit_list[each_commit_idx]
        this_entry = commit2codemsg.get(each_commit, {})
        diff = this_entry.get("diff", "")
        msg = this_entry.get("commit_msg", "")

        this_str = f"Commit message: {msg}\nDiff code: {diff}"
        this_str = re.sub(r"\s+", " ", this_str.replace("\n", " "))[:50000]

        token_count = vo.count_tokens(this_str, model="voyage-3")
        if sum(current_batch_tokens) + token_count <= voyage_max_tokens:
            current_batch.append((each_commit, this_str, each_commit_idx))
            current_batch_tokens.append(token_count)
        else:
            local_task_l.append(current_batch)
            local_total_token_count += sum(current_batch_tokens)
            current_batch = [(each_commit, this_str, each_commit_idx)]
            current_batch_tokens = [token_count]

    if len(current_batch) > 0 and sum(current_batch_tokens) + token_count <= voyage_max_tokens:
        local_task_l.append(current_batch)
        local_total_token_count += sum(current_batch_tokens)

    with task_l_lock:
        shared_task_l.extend(local_task_l)
    shared_total_token_count.value += local_total_token_count

def create_task_l(process_func, commit_list, commit2codemsg, MAX_PROCESSES=5):
    with multiprocessing.Manager() as manager:
       shared_task_l = manager.list()
       shared_total_token_count = manager.Value("i", 0)  # Shared counter
       task_l_lock = manager.Lock()

       chunk_size = len(commit_list) // MAX_PROCESSES
       if chunk_size == 0: return [], 0
       commit_chunks = [[x for x in range(i, min(i + chunk_size, len(commit_list)))] for i in range(0, len(commit_list), chunk_size)]

       with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
            list(tqdm.tqdm(pool.starmap(process_func,
                                        [(chunk, commit2codemsg, commit_list, shared_task_l, task_l_lock, shared_total_token_count) for chunk in commit_chunks]), total=len(commit_chunks)))

       task_l = list(shared_task_l)
       total_token_count = shared_total_token_count.value
    return task_l, total_token_count

def index_with_voyage(each_row):
   cve_list = list(each_row[1]["cve"])
   owner = list(each_row[1]["owner"])[0]
   repo = list(each_row[1]["repo"])[0]
   print(owner, repo, flush=True)
   repo_directory = feature_path + owner + "@@" + repo + "/"
   if os.path.exists(repo_directory + "/voyage/commit2embedding.json"):
      existing_commit2embedding = json.load(open(repo_directory + "/voyage/commit2embedding.json", "r"))
   else:
      existing_commit2embedding = {}
   print("place C", flush=True)

   #if os.path.exists(repo_directory + "/voyage/commit2embedding.json"): return
   if not os.path.exists(input_data_path + owner + "@@" + repo + ".json"):
       return 0
   try:
       commit2codemsg = json.load(open(input_data_path + owner + "@@" + repo + ".json", "r"))
   except json.decoder.JSONDecodeError:
       return 0

   commit_list = get_commit_list_test(feature_path, owner, repo, cve_list, BM25_K=10000)
   print("place B", flush=True)
   if len(existing_commit2embedding) >= len(commit_list) * 0.99:
       print("skip", owner, repo, len(existing_commit2embedding), len(commit_list), flush=True)
       return
   
   print("before", owner, repo, len(existing_commit2embedding), len(commit_list), flush=True)
   commit_list = list(set(commit_list).difference(existing_commit2embedding.keys()))
   print("after", len(commit_list), flush=True)

   task_l, total_token_count = create_task_l(process_commits_voyage, commit_list, commit2codemsg, MAX_PROCESSES=5)

   processes = 10
   with multiprocessing.Pool(processes=processes) as pool:
       with tqdm.tqdm(total=len(task_l), desc='Multiprocessing Example') as pbar:
           for result in pool.imap_unordered(process_task_voyage, task_l):
               existing_commit2embedding.update(result)
               pbar.update(1)
  
   if not os.path.exists(repo_directory + "/voyage/"):
       os.makedirs(repo_directory + "/voyage/")
   json.dump(existing_commit2embedding, open(repo_directory + "/voyage/commit2embedding.json", "w"))

def index_with_huggingface(repo2cve2negcommits, each_row, model, model_name, context_window, is_train = False):
    cve_list = list(each_row[1]["cve"])
    owner = list(each_row[1]["owner"])[0]
    repo = list(each_row[1]["repo"])[0]

    fout = open("error_log.txt", "w")

    print(owner, repo, flush=True)
    repo_directory = feature_path + owner + "@@" + repo + "/"
    if os.path.exists(repo_directory + "/" + model_name + "/commit2embedding.json"):
        existing_commit2embedding = json.load(open(repo_directory + "/" + model_name + "/commit2embedding.json", "r"))
    else:
        existing_commit2embedding = {}
    print("place C", flush=True)

    if not os.path.exists(input_data_path + owner + "@@" + repo + ".json"):
       return 0
    try:
        commit2codemsg = json.load(open(input_data_path + owner + "@@" + repo + ".json", "r"))
    except json.decoder.JSONDecodeError:
        return 0

    if is_train:
        commit_list = get_commit_list_train(repo2cve2negcommits, owner, repo, cve_list)
        if len(commit_list) == 0:
            return
        commit_list.update(each_row[1]["patch"].tolist())
        commit_list = list(set(commit_list))
        #raise Exception(len(commit_list), owner, repo)
        print("get commit list for ", owner, repo, flush=True)
        #raise Exception(len(set(commit_list).difference(existing_commitlist["commit_id"])))
        #print("skip", owner, repo, len(existing_commitlist), len(commit_list), flush=True)
        print("before", owner, repo, len(existing_commit2embedding), len(commit_list), flush=True)
        commit_list = list(set(commit_list).difference(existing_commit2embedding.keys()))
        print("after", len(commit_list), flush=True)
        if len(commit_list) == 0:
            return
    else:
        commit_list = get_commit_list_test(feature_path, owner, repo, cve_list)
        print("place B", flush=True)
        if len(existing_commit2embedding) >= len(commit_list) * 0.99:
            return
        print("before", owner, repo, len(existing_commit2embedding), len(commit_list), flush=True)
        commit_list = list(set(commit_list).difference(existing_commit2embedding.keys()))
        print("after", len(commit_list), flush=True)

    task_l = []
    documents = [] #open("commit_netty_89c241.txt").read()] #[

    task_l, total_token_count = create_task_l(process_commits_huggingface, commit_list, commit2codemsg, MAX_PROCESSES=5)
    
    for batch_id in tqdm.tqdm(range(len(task_l))):
        each_task_l = task_l[batch_id]
        result = process_task_huggingface(each_task_l, model, batch_id, fout, model_name, context_window)
        existing_commit2embedding.update(result)
       
    if not os.path.exists(repo_directory + "/" + model_name + "/"):
       os.makedirs(repo_directory + "/" + model_name + "/")
    json.dump(existing_commit2embedding, open(repo_directory + "/" + model_name + "/commit2embedding.json", "w"))

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   
   parser.add_argument("--model_name", type=str, default="grit_instruct_512_file")
   parser.add_argument("--dataset_name", type=str, default="AD")
   parser.add_argument("--is_train", action="store_true")

   args = parser.parse_args()

   model_name = args.model_name
   dataset_name = args.dataset_name
   is_train = args.is_train

   repo2cve2negcommits = json.load(open(feature_path + f"repo2cve2negcommits_{dataset_name}_500_unsampled.json" if args.dataset_name == "patchfinder" else f"../feature/repo2cve2negcommits_{dataset_name}_500.json", "r"))

   if args.model_name != "voyage":
      from transformers import AutoModel, AutoTokenizer
      from sentence_transformers import SentenceTransformer, util
      import torch
      import torch.nn.functional as F
      from torch import Tensor

   special_repos = []
   cve_data = read_csv(f"../csv/{dataset_name}_train.csv" if is_train else f"../csv/{dataset_name}_test.csv")
   groupby_list = sorted(list(cve_data.groupby(["owner", "repo"])), key = lambda x:x[0])

   unfinished = [x for x in range(len(groupby_list)) if not os.path.exists("../feature/" + groupby_list[x][0][0] + "@@" + groupby_list[x][0][1] + f"/{model_name}/commit2embedding.json")]

   groupby_list = [groupby_list[unfinished[x]] for x in range(0, len(unfinished))]

   if args.model_name == "voyage":
       
        #total_counts = {} 
        ##total_char_counts = []
        new_groupby_list = []
        for each_row in groupby_list:
            cve_list = list(each_row[1]["cve"])
            owner = list(each_row[1]["owner"])[0]
            repo = list(each_row[1]["repo"])[0]
            #if owner + "@@" + repo not in special_repos: continue
            new_groupby_list.append(each_row)
        for each_row in tqdm.tqdm(new_groupby_list):
            index_with_voyage(each_row)
   else:
        if model_name.startswith("grit"):
            from gritlm import GritLM
            import gritlm
            print(gritlm.__file__)
            model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")
        for each_row in tqdm.tqdm(groupby_list):
            if each_row[0] != ("xuxueli", "xxl-job"): continue
            index_with_huggingface(repo2cve2negcommits, each_row, model, model_name, 512, is_train = is_train)
