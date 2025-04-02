import os
import json
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm
from multiprocessing import Pool
import logging
import sys

dataset = sys.argv[1]
mode = sys.argv[2]

cve2desc_path = "../csv/cve2desc.json"
valid_list_path = f"../csv/{dataset}_{mode}.csv"

logging.basicConfig(level=logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
es = Elasticsearch("http://localhost:9200", http_compress=True, timeout=60, max_retries=3, retry_on_timeout=True)

with open(cve2desc_path, "r", encoding="utf-8") as f:
    cve2desc = json.load(f)


valid_list_df = pd.read_csv(valid_list_path)
valid_list_df = valid_list_df[(valid_list_df['repo'] == 'xxl-job') | (valid_list_df['repo'] == 'Valine')] # test
if mode == "train":
    INDEX_SUFFIX = "_file_train"
else:
    INDEX_SUFFIX = "_file"
    
    
def get_all_commit_features(cve_desc, index_name, scroll_time="2m", batch_size=10000):
    """
    Use the scroll API to extract all matching commit records from the specified index.
    Returns a list where each element is a dictionary containing commit_id, score, and file_name fields.
    """
    query = {
        "bool": {
            "should": [
                {"match_all": {}},
                {"multi_match": {"query": cve_desc, "fields": ["diff^0.5"]}}
            ]
        }
    }
    try:
        response = es.search(
            index=index_name,
            query=query,
            _source=["commit_id", "file_name"],
            size=batch_size,
            scroll=scroll_time,
            request_timeout=120
        )
    except Exception as e:
        print(f"Error searching index {index_name}: {e}")
        return []
    
    all_results = []
    scroll_id = response.get('_scroll_id')
    hits = response.get("hits", {}).get("hits", [])
    for hit in hits:
        all_results.append({
            "commit_id": hit["_source"].get("commit_id"),
            "score": hit.get("_score"),
            "file_name": hit["_source"].get("file_name")
        })
    while True:
        try:
            response = es.scroll(scroll_id=scroll_id, scroll=scroll_time, request_timeout=120)
        except Exception as e:
            print(f"Error scrolling index {index_name}: {e}")
            break
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            all_results.append({
                "commit_id": hit["_source"].get("commit_id"),
                "score": hit.get("_score"),
                "file_name": hit["_source"].get("file_name")
            })
        scroll_id = response.get("_scroll_id")
    try:
        es.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass
    return all_results

def group_commit_features(features):
    """
    Group records by commit_id, merging records of the same commit:
      - For each commit_id, generate a list of pairs in the format [file_name, score];
      - Sort the list in descending order by score;
    Return a dictionary where the key is commit_id and the value is the list of pairs.
    """
    grouped = {}
    for item in features:
        commit_id = item["commit_id"]
        if commit_id not in grouped:
            grouped[commit_id] = []
        grouped[commit_id].append([item.get("file_name"), item.get("score")])
    for commit_id in grouped:
        grouped[commit_id] = sorted(grouped[commit_id], key=lambda pair: pair[1] if pair[1] is not None else 0, reverse=True)
    return grouped

def process_cve(args):
    """
    For a given CVE:
      - Retrieve the CVE description from cve2desc.
      - Use the scroll API to fetch all matching commit records (keeping only commit_id, score, and file_name).
      - Merge records with the same commit_id and sort them by BM25 score in descending order.
      - For each commit, generate a file_id based on alphabetical sorting of filenames (starting from 0), without altering the BM25 order.
      - Finally, output the results in CSV format (commit_id, filename, bm25score, file_id) to the output_dir in a file named {cve_id}.csv.
    """
    row, output_dir = args
    cve_id = row["cve"]
    output_file = os.path.join(output_dir, f"{cve_id}.csv")
    # skip existing files or empty files
    if os.path.exists(output_file):
        if os.path.getsize(output_file) > 100:
            # print(f"Skipping {cve_id} as output file already exists and is not empty: {output_file}")
            return
        else:
            print(f"Output file {output_file} exists but is empty. Reprocessing {cve_id}...")
    owner = row["owner"]
    repo = row["repo"]
    index_name = f"{owner}@@{repo}{INDEX_SUFFIX}".lower()
    cve_desc = cve2desc.get(cve_id, [{"lang": "en", "value": ""}])[0]["value"]
    features = get_all_commit_features(cve_desc, index_name)
    grouped_features = group_commit_features(features)
    lines = []
    lines.append("commit_id,filename,bm25score,file_id")
    for commit_id, file_list in grouped_features.items():
        # Extract all filenames, sort them alphabetically, and generate a mapping from filename to file_id
        file_names = [pair[0] for pair in file_list if pair[0] is not None]
        sorted_file_names = sorted(file_names)
        file_id_mapping = {fname: idx for idx, fname in enumerate(sorted_file_names)}
        # Output lines in descending order of BM25 score without altering their order
        for pair in file_list:
            fname, score = pair
            file_id = file_id_mapping.get(fname, "")
            score_str = f"{score}" if score is not None else ""
            row_line = f"{commit_id},{fname},{score_str},{file_id}"
            lines.append(row_line)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Saved features for CVE {cve_id} to {output_file}")
    except Exception as e:
        print(f"Error saving features for {cve_id}: {e}")

def repo_already_processed(group, output_dir, threshold=100):
    """
    Check if all CVE files in the current repository exist and are non-empty (larger than threshold bytes).
    If all conditions are satisfied, return True, indicating that the repository has been processed; otherwise, return False.
    """
    for _, row in group.iterrows():
        output_file = os.path.join(output_dir, f"{row['cve']}.csv")
        if not os.path.exists(output_file) or os.path.getsize(output_file) <= threshold:
            return False
    return True


def process_all_cves_by_repo(df, num_processes=6):
    grouped = df.groupby(["owner", "repo"])
    for (owner, repo), group in grouped:
        output_dir = f"../feature/{owner}@@{repo}/bm25_files/result"
        os.makedirs(output_dir, exist_ok=True)
        if repo_already_processed(group, output_dir):
            print(f"Repo {owner}/{repo} already processed. Skipping.")
            continue
        

        # index_name = f"{owner}@@{repo}_file".lower()
        # try:
        #     es.indices.open(index=index_name)
        #     # print(f"Opened index {index_name} for {owner}/{repo}.")
        # except Exception as e:
        #     print(f"Index {index_name} might already be open or cannot be opened: {e}")
        
        print(f"Processing CVEs for {owner}/{repo} ...")
        tasks = [(row, output_dir) for _, row in group.iterrows()]
        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap(process_cve, tasks), total=len(tasks), desc=f"Processing {owner}/{repo}"))

        # try:
        #     es.indices.close(index=index_name)
        #     print(f"Closed index {index_name} for {owner}/{repo}.")
        # except Exception as e:
        #     print(f"Error closing index {index_name}: {e}")

if __name__ == "__main__":
    process_all_cves_by_repo(valid_list_df, num_processes=8)
