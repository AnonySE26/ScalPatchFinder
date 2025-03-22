import os
import json
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch
from tqdm import tqdm
from multiprocessing import Pool
import sys

dataset = sys.argv[1]

MAX_RANK = 100000
cve2desc_path = "../csv/cve2desc.json"
# output_dir = "./patchfinder_ranked_commits_bm25_time"
valid_list_path = f"../csv/{dataset}.csv"
output_dir = f"./{dataset}_ranked_commits_bm25_time"
es = Elasticsearch("http://localhost:9200")


with open(cve2desc_path, "r") as f:
    cve2desc = json.load(f)

valid_list_df = pd.read_csv(valid_list_path)
valid_list_df["publish_time"] = pd.to_datetime(valid_list_df["publish_time"], errors="coerce")
valid_list_df["reserve_time"] = pd.to_datetime(valid_list_df["reserve_time"], errors="coerce")
# valid_list_df = valid_list_df[valid_list_df["repo"] == "tensorflow"]
EXCLUDED_REPOS = {"linux", "tomcat","aws-sdk-java", "libpod"}
valid_list_df = valid_list_df[~valid_list_df["repo"].isin(EXCLUDED_REPOS)]


os.makedirs(output_dir, exist_ok=True)


def get_ranked_commit_list(cve_desc, index_name, repo, MAX_RANK=50000):
    response = es.search(
        index=index_name,
        query={
            "bool": {
                "should": [
                    {"match_all": {}}, 
                    {
                        "multi_match": {
                            "query": cve_desc,
                            "fields": ["commit_msg^0.7", "diff^0.3"]
                        }
                    }
                ]
            }
        },
        _source=["commit_id", "datetime"],
        size=MAX_RANK,
        request_timeout=60
    )
    hits = response.get("hits", {}).get("hits", [])
    # print(f"Found {len(hits)} hits for {cve_desc} in {repo}.")
    return [
        {
            "commit_id": hit["_source"]["commit_id"],
            "datetime": hit["_source"]["datetime"],
            "score": hit["_score"],
            "repo": repo
        }
        for hit in hits
    ]

def process_repo(repo, group_df):
    owner = group_df.iloc[0]["owner"]
    index_name = f"{owner}@@{repo}".lower()
    results = []
    try:
        es.indices.open(index=index_name, wait_for_active_shards=1, request_timeout=60)
    except Exception as e:
        print(f"Error opening index {index_name}: {e}")
        # If opening the index fails, mark all CVEs for this repository as unprocessed.
        for _, row in group_df.iterrows():
            cve_id = row["cve"]
            results.append((cve_id, None, None, None))
        return results

    # increse max_result_window
    es.indices.put_settings(
        index=index_name,
        body={"index": {"max_result_window": MAX_RANK}}
    )

    for _, row in group_df.iterrows():
        cve_id = row["cve"]
        patch = row["patch"]
        cve_file_path = os.path.join(output_dir, f"{cve_id}.json")
        if os.path.exists(cve_file_path):
            print(f"Skipping {cve_id} as it is already processed.")
            results.append((cve_id, None, None, None))
            continue

        cve_desc = cve2desc.get(cve_id, [{"lang": "en", "value": ""}])[0]["value"]
        commits = get_ranked_commit_list(cve_desc, index_name, repo, MAX_RANK=MAX_RANK)

        # Calculate the ranking and BM25 score corresponding to the patch
        rank = -1
        score = None
        for rank_idx, commit in enumerate(commits, start=1):
            if commit["commit_id"].strip() == patch.strip():
                rank = rank_idx
                score = commit["score"]
                break

        publish_time = row["publish_time"]
        reserve_time = row["reserve_time"]

        # Sort commits by commit time
        try:
            commits.sort(key=lambda x: datetime.fromisoformat(x["datetime"]))
        except Exception as e:
            print(f"Error sorting commits for CVE {cve_id} in repo {repo}: {e}")

        publish_commit_idx, reserve_commit_idx = None, None
        min_publish_diff, min_reserve_diff = float("inf"), float("inf")
        for idx, commit in enumerate(commits):
            commit_time = datetime.fromisoformat(commit["datetime"])
            if publish_time:
                diff = abs((commit_time - publish_time).total_seconds())
                if diff < min_publish_diff:
                    min_publish_diff = diff
                    publish_commit_idx = idx
            if reserve_time:
                diff = abs((commit_time - reserve_time).total_seconds())
                if diff < min_reserve_diff:
                    min_reserve_diff = diff
                    reserve_commit_idx = idx

        closest_commit_info = {
            "cve": cve_id,
            "publish_commit_id": commits[publish_commit_idx]["commit_id"] if publish_commit_idx is not None else None,
            "publish_commit_datetime": commits[publish_commit_idx]["datetime"] if publish_commit_idx is not None else None,
            "reserve_commit_id": commits[reserve_commit_idx]["commit_id"] if reserve_commit_idx is not None else None,
            "reserve_commit_datetime": commits[reserve_commit_idx]["datetime"] if reserve_commit_idx is not None else None
        }

        for idx, commit in enumerate(commits):
            commit["publish_diff"] = idx - publish_commit_idx if publish_commit_idx is not None else None
            commit["reserve_time_diff"] = idx - reserve_commit_idx if reserve_commit_idx is not None else None

        with open(cve_file_path, "w") as f:
            json.dump(commits, f, indent=4)
        results.append((cve_id, rank, score, closest_commit_info))
    
    try:
        es.indices.close(index=index_name)
    except Exception as e:
        print(f"Error closing index {index_name}: {e}")
    return results


def process_cve(row):
    cve_id = row["cve"]
    patch = row["patch"]
    repo = row["repo"]
    index_name = f"{row['owner']}@@{repo}".lower()
    cve_desc = cve2desc.get(cve_id, [{"lang": "en", "value": ""}])[0]["value"]
    
    cve_file_path = os.path.join(output_dir, f"{cve_id}.json")
    if os.path.exists(cve_file_path):
        print(f"Skipping {cve_id} as it is already processed.")
        return cve_id, None, None, None
    
    try:
        es.indices.open(index=index_name, wait_for_active_shards=1)
    except Exception as e:
        print(f"Error opening index {index_name}: {e}")
        return cve_id, None, None, None
    
    es.indices.put_settings(
        index=index_name,
        body={"index": {"max_result_window": MAX_RANK}}
    )

    commits = get_ranked_commit_list(cve_desc, index_name, repo, MAX_RANK=MAX_RANK)
    # print(f"Found {len(commits)} commits for {cve_id} in {repo}.")

    cve_file_path = os.path.join(output_dir, f"{cve_id}.json")

    rank = -1
    score = None
    for rank_idx, commit in enumerate(commits, start=1):
        if commit["commit_id"].strip() == patch.strip():
            rank = rank_idx
            score = commit["score"]
            break

    publish_time = row["publish_time"]
    reserve_time = row["reserve_time"]

    # find closest commit
    publish_commit_idx, reserve_commit_idx = None, None
    min_publish_diff, min_reserve_diff = float("inf"), float("inf")

    commits.sort(key=lambda x: datetime.fromisoformat(x["datetime"]))

    for idx, commit in enumerate(commits):
        commit_time = datetime.fromisoformat(commit["datetime"])

        if publish_time:
            diff = abs((commit_time - publish_time).total_seconds())
            if diff < min_publish_diff:
                min_publish_diff = diff
                publish_commit_idx = idx

        if reserve_time:
            diff = abs((commit_time - reserve_time).total_seconds())
            if diff < min_reserve_diff:
                min_reserve_diff = diff
                reserve_commit_idx = idx

    closest_commit_info = {
        "cve": cve_id,
        "publish_commit_id": commits[publish_commit_idx]["commit_id"] if publish_commit_idx is not None else None,
        "publish_commit_datetime": commits[publish_commit_idx]["datetime"] if publish_commit_idx is not None else None,
        "reserve_commit_id": commits[reserve_commit_idx]["commit_id"] if reserve_commit_idx is not None else None,
        "reserve_commit_datetime": commits[reserve_commit_idx]["datetime"] if reserve_commit_idx is not None else None
    }

    # Calculate the position difference for each commit
    for idx, commit in enumerate(commits):
        commit["publish_diff"] = idx - publish_commit_idx if publish_commit_idx is not None else None
        commit["reserve_time_diff"] = idx - reserve_commit_idx if reserve_commit_idx is not None else None

    with open(cve_file_path, "w") as f:
        json.dump(commits, f, indent=4)
        
    try:
        es.indices.close(index=index_name)
    except Exception as e:
        print(f"Error closing index {index_name}: {e}")

    return cve_id, rank, score, closest_commit_info

def process_repo_helper(args):
    return process_repo(*args)

def process_all_repos(df, num_processes=4):
    """
    First, sort by repo and group them, then process each repo group in parallel.
    """
    df_sorted = df.sort_values(by="repo")
    grouped = df_sorted.groupby("repo")
    repo_groups = [(repo, group.copy()) for repo, group in grouped]

    all_results = []
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_repo_helper, repo_groups),
                        total=len(repo_groups),
                        desc="Processing repos"):
            all_results.extend(result)

    ranks = []
    scores = []
    closest_commits = []
    for cve_id, rank, score, closest_commit_info in all_results:
        if rank is not None:
            ranks.append(rank)
            scores.append(score)
            closest_commits.append(closest_commit_info)
    return ranks, scores, closest_commits

def process_all_cves(df, num_processes=4):
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_cve, [row for _, row in df.iterrows()]), total=len(df), desc="Processing CVEs"))
    
    ranks = []
    scores = []
    closest_commits = []

    for cve_id, rank, score, closest_commit_info in results:
        if rank is not None: 
            ranks.append(rank)
            scores.append(score)
            closest_commits.append(closest_commit_info)

    return ranks, scores, closest_commits

if __name__ == "__main__":
    num_processes = 8
    # ranks, scores, closest_commits = process_all_cves(valid_list_df, num_processes=num_processes)
    ranks, scores, closest_commits = process_all_repos(valid_list_df, num_processes=num_processes)