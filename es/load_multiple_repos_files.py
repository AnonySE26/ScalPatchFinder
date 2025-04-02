#!/usr/bin/env python3
import os
import json
import logging
import time
import re
from tqdm import tqdm
import pandas as pd
from dateutil import parser
from elasticsearch import Elasticsearch, helpers
from multiprocessing import Pool
import argparse


logging.basicConfig(level=logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
es = Elasticsearch("http://localhost:9200", http_compress=True, timeout=60, max_retries=3, retry_on_timeout=True)


BASE_DIRECTORY = "../repo2commits_diff"


PROCESSED_INDEXES_FILE = "processed_indexes_file.txt"
if not os.path.exists(PROCESSED_INDEXES_FILE):
    with open(PROCESSED_INDEXES_FILE, "w") as f:
        pass


RETRY_FAILED_INDEXES_FILE = "retry_failed_indexes.txt"
if not os.path.exists(RETRY_FAILED_INDEXES_FILE):
    with open(RETRY_FAILED_INDEXES_FILE, "w") as f:
        pass

def save_retry_failed_index(index_name):
    with open(RETRY_FAILED_INDEXES_FILE, "a") as f:
        f.write(f"{index_name}\n")

def get_processed_indexes():
    with open(PROCESSED_INDEXES_FILE, "r") as f:
        return set(line.strip() for line in f)

def save_processed_index(index_name):
    with open(PROCESSED_INDEXES_FILE, "a") as f:
        f.write(f"{index_name}\n")


def create_index(index_name):
    if not es.indices.exists(index=index_name):
        body = {
            "settings": {
                "analysis": {
                    "filter": {
                        "my_word_delimiter": {
                            "type": "word_delimiter_graph",
                            "split_on_case_change": True,
                            "split_on_symbols": True,
                            "split_on_numerics": True,
                            "preserve_original": True
                        }
                    },
                    "analyzer": {
                        "my_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "my_word_delimiter",
                                "lowercase"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "commit_id": {"type": "keyword"},
                    "datetime": {"type": "date"},
                    "commit_msg": {"type": "text", "analyzer": "my_analyzer"},
                    "diff": {"type": "text", "analyzer": "my_analyzer"},
                    "repo": {"type": "keyword"},
                    "file_name": {"type": "keyword"}
                }
            }
        }
        es.indices.create(index=index_name, body=body)
    else:
        try:
            es.indices.open(index=index_name)
        except Exception as e:
            print(f"Failed to open index {index_name}: {e}")

# Split the diff by file, returning a dict: {file_name: diff_chunk}
def split_diff_into_files(diff_text):
    # Split using "diff --git"
    chunks = [chunk for chunk in diff_text.split("diff --git") if chunk.strip()]
    file_chunks = {}
    for chunk in chunks:
        lines = chunk.splitlines()
        if not lines:
            continue
        # Extract the file path after "a/" as the file name
        header = lines[0]
        m = re.search(r'a/(\S+)', header)
        file_name = m.group(1) if m else "unknown"
        file_chunks[file_name] = "diff --git" + chunk  # Restore the missing prefix
    return file_chunks

# For a single commit record, split the diff into multiple records (one per file)
def split_commit_record(record, max_length=50):
    if "diff" in record and isinstance(record["diff"], str):
        diff_text = record["diff"]
        file_chunks = split_diff_into_files(diff_text)
        if file_chunks:
            new_records = []
            # After sorting file_name alphabetically, keep only the first 50 files
            for file_name in sorted(file_chunks.keys())[:max_length]:
                new_rec = record.copy()
                new_rec["diff"] = file_chunks[file_name]
                new_rec["file_name"] = file_name
                new_records.append(new_rec)
            return new_records
    # If there's no diff or splitting fails, return the original record and set file_name to None
    new_rec = record.copy()
    new_rec["file_name"] = None
    return [new_rec]


def fix_datetime_string(dt_str):
    m = re.search(r'([+-]\d{5})$', dt_str)
    if m:
        tz = m.group(1)
        sign = tz[0]
        hh = tz[1:3]
        mm = tz[3:5]
        new_tz = f"{sign}{hh}:{mm}:00"
        dt_str = dt_str[:-len(tz)] + new_tz
    return dt_str

# Convert the datetime field to ISO format
def convert_datetime_field(record):
    if "datetime" in record:
        original = record["datetime"]
        try:
            fixed = fix_datetime_string(original)
            dt = parser.parse(fixed)
            record["datetime"] = dt.isoformat()
        except Exception as e:
            print(f"Failed to parse datetime '{original}': {e}")
    return record

# Prepare the bulk data for Elasticsearch
def prepare_bulk_data(json_data, index_name, batch_size=100):
    docs = []
    for record in json_data:
        record = convert_datetime_field(record)
        split_records = split_commit_record(record)
        docs.extend(split_records)
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        actions = []
        for rec in batch:
            commit_id = rec.get("commit_id", "")
            file_name = rec.get("file_name")
            # Use a combination of commit_id and file_name as _id to avoid duplicates for the same file in the same commit
            if file_name is not None:
                doc_id = f"{commit_id}_{file_name}"
            else:
                doc_id = commit_id
            actions.append({"_index": index_name, "_id": doc_id, "_source": rec})
        yield actions


def import_split_files(directory, index_name, batch_size=100, allowed_commits=None):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")])
    total_records = 0
    for file in tqdm(files, desc=f"Processing files in {index_name}"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error in file {file}: {e}")
            continue
        # If allowed_commits is specified (e.g., in train mode), filter commit_id
        if allowed_commits is not None:
            data = [record for record in data if record.get("commit_id") in allowed_commits]
        for batch in prepare_bulk_data(data, index_name, batch_size=batch_size):
            retries = 0
            while retries < 10:
                try:
                    helpers.bulk(es, batch)
                    break
                except Exception as e:
                    retries += 1
                    # print(f"Error importing batch from {file}, retry {retries}/{10}: {e}")
                    time.sleep(15)
            if retries >= 10:
                save_retry_failed_index(index_name)
            time.sleep(1)
        total_records += len(data)
    return total_records

def process_repo(args):
    if len(args) == 3:
        owner, repo, allowed_commits = args
        index_suffix = "_file_train"
    else:
        owner, repo = args
        allowed_commits = None
        index_suffix ="_file"
        
    index_name = f"{owner}@@{repo}".lower() + index_suffix
    processed_indexes = get_processed_indexes()
    if index_name in processed_indexes:
        print(f"Skipping already processed index: {index_name}")
        return

    repo_dir_name = f"split_{owner}@@{repo}"
    repo_dir = os.path.join(BASE_DIRECTORY, repo_dir_name)
    if not os.path.isdir(repo_dir):
        print(f"Directory not found: {repo_dir}")
        return

    print(f"Processing repo {owner}@@{repo} -> index: {index_name}")
    create_index(index_name)
    total_records = import_split_files(repo_dir, index_name, batch_size=500, allowed_commits=allowed_commits)
    doc_count = es.count(index=index_name)['count']
    print(f"Index '{index_name}': imported {total_records} commits, ES count after split by files: {doc_count}")
    save_processed_index(index_name)

def main():
    parser = argparse.ArgumentParser(description="Import split commit diffs into Elasticsearch.")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., AD, patchfinder")

    args = parser.parse_args()
    mode = args.mode
    dataset = args.dataset
    
    num_processes = 1
    global PROCESSED_INDEXES_FILE
    if mode == "train":

        PROCESSED_INDEXES_FILE = f"processed_indexes_file.txt"
        train_file = f"../feature/repo2commits_{dataset}_500.json"
        with open(train_file, "r") as f:
            train_data = json.load(f)
        # train_data  { "owner@@repo": [commit_id, ...], ... }
        repo_list = []
        for key, commit_ids in train_data.items():
            if "@@" in key:
                owner, repo = key.split("@@", 1)
            else:
                owner = None
                repo = key
            repo_list.append((owner, repo, commit_ids))
        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap(process_repo, repo_list), total=len(repo_list), desc="Processing Repos (train)"))
    else:
        CSV_FILE = f"../csv/{dataset}_{mode}.csv"
        df = pd.read_csv(CSV_FILE)
        unique_repos = df[['owner', 'repo']].drop_duplicates()
        repo_list = [(row['owner'], row['repo']) for _, row in unique_repos.iterrows()]
        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap(process_repo, repo_list), total=len(repo_list), desc="Processing Repos"))

if __name__ == "__main__":
    main()
