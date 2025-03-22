import os
import json
import logging
import time
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from dateutil import parser
import re


logging.basicConfig(level=logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

es = Elasticsearch("http://localhost:9200", http_compress=True, timeout=60, max_retries=3, retry_on_timeout=True)


BASE_DIRECTORY = "./repo2commits_diff"
EXCLUDED_REPOS = {"torvalds@@linux", "aws@@aws-sdk-java", "mjg59@@linux", "containers@@libpod"}
PROCESSED_INDEXES_FILE = "processed_indexes.txt"


if not os.path.exists(PROCESSED_INDEXES_FILE):
    with open(PROCESSED_INDEXES_FILE, "w") as f:
        pass


def get_processed_indexes():
    with open(PROCESSED_INDEXES_FILE, "r") as f:
        return set(line.strip() for line in f)

def save_processed_index(index_name):
    with open(PROCESSED_INDEXES_FILE, "a") as f:
        f.write(f"{index_name}\n")


def create_index(index_name):
    if not es.indices.exists(index=index_name):
        es.indices.create(
            index=index_name,
            body={
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
                        "repo": {"type": "keyword"} 
                    }
                }
            }
        )


def prepare_bulk_data(json_data, index_name, batch_size=100):
    for i in range(0, len(json_data), batch_size):
        batch = json_data[i:i + batch_size]
        actions = [
            {"_index": index_name, "_source": record} for record in batch
        ]
        yield actions


def import_split_files(directory, index_name, batch_size=100):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")])
    total_records = 0

    for file in tqdm(files, desc=f"Processing files in {index_name}"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error in file {file}: {e}")
            continue

        # truncate diff
        data = [convert_datetime_field(truncate_diff(record, max_length=25000)) for record in data]

        for batch in prepare_bulk_data(data, index_name, batch_size=batch_size):
            try:
                helpers.bulk(es, batch)
            except Exception as e:
                print(f"Error importing batch from {file}: {e}")
                time.sleep(10)
                try:
                    helpers.bulk(es, batch)
                except Exception as retry_e:
                    print(f"Retry failed for batch from {file}: {retry_e}")
            time.sleep(1) 
        total_records += len(data)

    return total_records


def truncate_diff(record, max_length=25000):
    if "diff" in record and isinstance(record["diff"], str):
        if len(record["diff"]) > max_length:
            record["diff"] = record["diff"][:max_length] + "... (truncated)"
    return record

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


def single_process_repos(base_directory):
    processed_indexes = get_processed_indexes()
    repo_directories = [
        d for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d)) and d.startswith("split_") and d not in EXCLUDED_REPOS
    ]

    for repo_dir in repo_directories:
        index_name = repo_dir.replace("split_", "").lower()

        if index_name in processed_indexes:
            # print(f"Skipping already processed index: {index_name}")
            continue

        create_index(index_name)

        repo_path = os.path.join(base_directory, repo_dir)
        total_records = import_split_files(repo_path, index_name, batch_size=10000)

        doc_count = es.count(index=index_name)['count']
        print(f"Total documents in index '{index_name}': {doc_count} (Expected: {total_records})")

        save_processed_index(index_name)


from multiprocessing import Pool

def process_single_repo(repo_dir):

    index_name = repo_dir.replace("split_", "").lower() 

    if index_name in get_processed_indexes() or index_name in EXCLUDED_REPOS:
        # print(f"Skipping already processed index: {index_name}")
        return

    create_index(index_name)

    repo_path = os.path.join(BASE_DIRECTORY, repo_dir)
    total_records = import_split_files(repo_path, index_name, batch_size=1000)

    doc_count = es.count(index=index_name)['count']
    print(f"Total documents in index '{index_name}': {doc_count} (Expected: {total_records})")

    save_processed_index(index_name)


def multi_process_repos(base_directory, num_workers=4):

    repo_directories = [
        d for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d)) and d.startswith("split_") and d not in EXCLUDED_REPOS
    ]

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_single_repo, repo_directories), total=len(repo_directories), desc="Processing Repos"))


if __name__ == "__main__":
    mode = "multi"
    if mode == "single":
        print("Running in single-process mode...")
        single_process_repos(BASE_DIRECTORY)
    elif mode == "multi":
        print("Running in multi-process mode...")
        multi_process_repos(BASE_DIRECTORY, num_workers=4)
