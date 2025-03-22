#!/bin/python
import re
#import tiktoken
import tqdm
import os
import json
import subprocess
import sys
import multiprocessing
import pandas
from pandas import read_csv
import numpy as np
from utils import get_commit_list_test, get_commit_list_train
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"


linecount = 0
import time

t1 = time.time()

grit_batch_size = 10 #int(sys.argv[4])
#context_window = 256 #int(sys.argv[4])
feature_path = "../../../feature/" # if len(sys.argv[1]) > 0 else ""
input_data_path = "../../../repo2commits_diff/" # if len(sys.argv[1]) > 0 else ""
INSTRUCTION = "This is a commit (commit message + diff code) of a repository. Represent it to retrieve the patching commit for a CVE description. "
FILE_TOPK = 30

 
def gritlm_instruction(INSTRUCTION):
    #instruction = "Given the description of a security vulnerability, retrieve the patching commit (commit message + diff code)."
    return "<|user|>\n" + INSTRUCTION + "\n<|embed|>\n" if INSTRUCTION else "<|embed|>\n"

def process_task_huggingface(task):
    #global model, device
    #import torch

    batch_id, model_name, context_window, model, this_batch = task
    documents = [x[1] for x in this_batch]
    commits = [x[0] for x in this_batch]

    commitid_list = []
    embedding_list = []

    try:

        documents_embeddings = model.encode(documents, instruction=gritlm_instruction(INSTRUCTION), batch_size = 4, max_length = context_window)

        for x in range(len(commits)):
            this_commit_id, this_filename = commits[x].split("@@")
            commitid_list.append({"commit_id": this_commit_id, "filename": this_filename})
            embedding_list.append(documents_embeddings[x].tolist())

        #print("place F", "device", device, flush=True)
        return commitid_list, embedding_list

    except Exception as e:
        import traceback
        error_msg = f"Batch {batch_id} Error: {str(e)}\nTraceback:\n{traceback.format_exc()}\n"
        
        print(error_msg, file=sys.stderr)
        sys.stderr.flush()
        with open("error_log.txt", "a") as fout:
            fout.write(error_msg)
            fout.flush()
        return [], []


def process_and_split_diff(diff, owner, repo, commit_id, FILE_TOPK=50):
    diff = re.sub("\s+", " ", diff.replace("\n", " "))
    #print(commit_id, diff, flush=True)
    diff_each_file = re.split("diff --git a/", diff)[1:]
    this_commit_batch = []
    filename2content = {}
    file_id = 0
    for each_file in diff_each_file:
        each_file_split = each_file.split()
        this_filename = each_file_split[0]
        this_filename2 = each_file_split[1]
        #if "b/" + this_filename != this_filename2: continue
        filename2content[this_filename] = each_file
    
    sorted_topk_filenames = sorted(filename2content.items(), key = lambda x:x[0])[:FILE_TOPK]
    for this_filename, this_file_content in sorted_topk_filenames:
        this_commit_batch.append((this_filename, this_file_content))
    return this_commit_batch

def process_commits_huggingface(owner, repo, chunk, commit2codemsg, commit_list, shared_task_l, task_l_lock, shared_total_token_count):
    local_task_l = []
    current_batch = []
    for each_commit_idx in tqdm.tqdm(chunk):
        each_commit = commit_list[each_commit_idx]
        this_entry = commit2codemsg.get(each_commit, {})
        diff_batch = process_and_split_diff(this_entry.get("diff", ""), owner, repo, each_commit)
        msg = this_entry.get("commit_msg", "")

        for (each_filename, each_diff_file) in diff_batch:
            this_str = f"Commit message: {msg}\nDiff code: {each_diff_file}"

            if len(current_batch) < grit_batch_size:
                current_batch.append((each_commit + "@@" + each_filename, this_str, each_commit_idx))
            else:
                local_task_l.append(current_batch)
                current_batch = [(each_commit + "@@" + each_filename, this_str, each_commit_idx)]
    if len(current_batch) > 0:
        local_task_l.append(current_batch)

    with task_l_lock:
        shared_task_l.extend(local_task_l)

def create_task_l(owner, repo, process_func, commit_list, commit2codemsg, MAX_PROCESSES=5):
    with multiprocessing.Manager() as manager:
       shared_task_l = manager.list()
       shared_total_token_count = manager.Value("i", 0)  # Shared counter
       task_l_lock = manager.Lock()

       chunk_size = len(commit_list) // MAX_PROCESSES
       if chunk_size == 0: return
       commit_chunks = [[x for x in range(i, min(i + chunk_size, len(commit_list)))] for i in range(0, len(commit_list), chunk_size)]

       with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
            list(tqdm.tqdm(pool.starmap(process_func,
                                        [(owner, repo, chunk, commit2codemsg, commit_list, shared_task_l, task_l_lock, shared_total_token_count) for chunk in commit_chunks]), total=len(commit_chunks)))

       task_l = list(shared_task_l)
       total_token_count = shared_total_token_count.value
    return task_l, total_token_count

def index_with_huggingface(repo2cve2negcommits, each_row, model_name, context_window, model, is_train=False):
    import glob
    cve_list = list(each_row[1]["cve"])
    owner = list(each_row[1]["owner"])[0]
    repo = list(each_row[1]["repo"])[0]

    fout = open("error_log.txt", "w")

    print(owner, repo, flush=True)
    repo_directory = feature_path + owner + "@@" + repo + "/"

    existing_files = glob.glob(repo_directory + "/" + model_name + "/commit_list_*.csv")
    if len(existing_files) > 0:
        existing_commitlist = []
        for each_file in existing_files:
            existing_commitlist.extend(read_csv(each_file).to_dict("records"))
        existing_commitlist = pandas.DataFrame(existing_commitlist)
        print("number of existing commit list", len(existing_commitlist), flush=True)

    else:
        existing_commitlist = pandas.DataFrame(columns=["commit_id", "filename"])
        existing_embeddings = []
    print("finished loading embedding for", owner, repo, flush=True)

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
        print("before", owner, repo, len(existing_commitlist), len(commit_list), flush=True)
        commit_list = list(set(commit_list).difference(existing_commitlist["commit_id"].tolist()))
        print("after", len(commit_list), flush=True)
        if len(commit_list) < 100:
            return
    else:
        commit_list = get_commit_list_test(feature_path, owner, repo, cve_list)
        print("get commit list for ", owner, repo, flush=True)
        if len(set(existing_commitlist["commit_id"])) >= len(commit_list) * 0.99:
            print("skip", owner, repo, len(existing_commitlist), len(commit_list), flush=True)
            return
        print("before", owner, repo, len(existing_commitlist), len(commit_list), flush=True)
        commit_list = list(set(commit_list).difference(existing_commitlist["commit_id"].tolist()))
        print("after", len(commit_list), flush=True)

    #existing_commitlist = existing_commitlist.to_dict("records")

    task_l = []
    documents = [] #open("commit_netty_89c241.txt").read()] #[

    task_l, total_token_count = create_task_l(owner, repo, process_commits_huggingface, commit_list, commit2codemsg, MAX_PROCESSES=5)

    task_l = [(x, model_name, context_window, model) + (task_l[x],) for x in range(len(task_l))]

    existing_commitlist = []
    existing_embeddings = []
    
    for batch_id in tqdm.tqdm(range(len(task_l))):
        each_task_l = task_l[batch_id]
        commitid_list, embedding_list = process_task_huggingface(each_task_l) # model.encode for the batches in each_task_l
        existing_commitlist.extend(commitid_list)
        existing_embeddings.extend(embedding_list)
       
    if not os.path.exists(repo_directory + "/" + model_name + "/"):
       os.makedirs(repo_directory + "/" + model_name + "/")

    try:
        existing_commitlist, existing_embeddings = rerank_and_add_fileid(pandas.DataFrame(existing_commitlist), existing_embeddings)

        # index too big, split index so each json file only stores 30000 files
        split_by_k = 30000
        print("length of existing commit list", len(existing_commitlist), flush=True)

        existing_files = glob.glob(repo_directory + "/" + model_name + "/commit_list_*.csv")
        max_suffix = -1
        for each_file in existing_files:
            this_suffix = int(each_file.split("_")[-1][:-4])
            max_suffix = max(max_suffix, this_suffix)

        if max_suffix == -1:
            offset = 0
        else:
            offset = max_suffix + split_by_k

        for i in range(0, len(existing_commitlist), split_by_k):
            existing_commitlist[i:i+split_by_k].to_csv(repo_directory + "/" + model_name + "/commit_list_" + str(offset + i) + ".csv")
            json.dump(existing_embeddings[i:i+split_by_k], open(repo_directory + "/" + model_name + "/embedding_list_" + str(offset + i) + ".json", "w"))
    except Exception as e:
        print("error in rerank_and_add_fileid", str(e), flush=True)
        fout.write(str(e))
        fout.flush()

def rerank_and_add_fileid(existing_commitlist, existing_embeddings):
    existing_commitlist["original_index"] = existing_commitlist.index
    df_sorted = existing_commitlist.sort_values(["commit_id", "filename"], ascending=[True, True]).reset_index(drop=True)
    df_sorted["file_id"] = df_sorted.groupby("commit_id").cumcount()

    existing_embeddings = [existing_embeddings[x] for x in df_sorted["original_index"].tolist()]
    return df_sorted, existing_embeddings


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--model_name", type=str, default="grit_instruct_512_file")
   parser.add_argument("--dataset_name", type=str, default="AD")
   parser.add_argument("--is_train", type=bool, action="store_false")

   args = parser.parse_args()

   model_name = args.model_name
   dataset_name = args.dataset_name
   context_window = 512 
   is_train = args.is_train

   raise Exception(model_name, dataset_name, is_train)

   train_K = 500
   import glob

   from transformers import AutoModel, AutoTokenizer
   from sentence_transformers import SentenceTransformer, util
   import torch
   import torch.nn.functional as F
   from torch import Tensor

   special_repos = []
   cve_data = read_csv(f"../csv/{dataset_name}_train.csv" if is_train else f"../csv/{dataset_name}_test.csv")
   groupby_list = sorted(list(cve_data.groupby(["owner", "repo"])), key = lambda x:x[0])


#    unfinished = [x for x in range(len(groupby_list)) if not glob.glob(feature_path + groupby_list[x][0][0] + "@@" + groupby_list[x][0][1] + f"/{model_name}/commit_list_*.csv")]
#    raise Exception(unfinished)

   unfinished = [x for x in range(len(groupby_list)) if groupby_list[x][0] == ("protobuffers", "protobuf")]
   groupby_list = [groupby_list[unfinished[x]] for x in range(len(unfinished) // 2, 3 * len(unfinished) // 4)]

   #raise Exception(groupby_list)

#    groupby_list = [groupby_list[x] for x in range(len(groupby_list)) \
# #                    #if os.path.exists(feature_path + groupby_list[x][0][0] + "@@" + groupby_list[x][0][1] + "/bm25_time/result/")][13:]
#                     if groupby_list[x][0] in [("ImageMagick", "ImageMagick6")]]
                   
                   
   #raise Exception([groupby_list[x][0] for x in range(45, 74)])

   repo2cve2negcommits = json.load(open(feature_path + f"repo2cve2negcommits_{dataset_name}_500_unsampled.json" if dataset_name == "patchfinder" else f"../../../feature/repo2cve2negcommits_{dataset_name}_500.json", "r"))

   if model_name.startswith("grit"):
       from gritlm import GritLM
       import gritlm
       print(gritlm.__file__)
       model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")

   # num_gpus = 1
   # gpu_ids = [0]

   for each_row in tqdm.tqdm(groupby_list):
       #if each_row[0] != ("php", "php-src"): continue
       index_with_huggingface(repo2cve2negcommits, each_row, model_name, context_window, model, is_train=is_train)
