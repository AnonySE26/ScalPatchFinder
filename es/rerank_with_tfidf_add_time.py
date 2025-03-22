import os
import json
import sys
from multiprocessing import Pool, cpu_count
import pandas as pd

# python rerank_with_tfidf_add_time.py 1.0 0.0 0.0 AD
if len(sys.argv) != 5:
    print("Usage: python rerank_with_tfidf_add_time.py <bm25_weight> <reserve_time_weight> <publish_time_weight> <dataset>")
    sys.exit(1)

bm25_weight = float(sys.argv[1])
reserve_time_weight = float(sys.argv[2])
publish_time_weight = float(sys.argv[3])
dataset = sys.argv[4]

input_dir = "./GithubAD_ranked_commits_bm25_time"
output_dir = "./GithubAD_ranked_commits_reranked"


valid_list_path = f"../csv/{dataset}.csv"
os.makedirs(output_dir, exist_ok=True)

# k_values = [1, 10, 100, 500, 1000, 2000, 3000, 4000, 5000]
k_values = [100, 500, 1000, 2000, 5000]

def reciprocal_rank(commits):
    """
    Recalculates BM25 and other scores' ranking scores using the method of 1 / rank.
    """
    commits.sort(key=lambda x: x.get("score", 0), reverse=True)  # Sort by BM25 in descending order
    for rank, commit in enumerate(commits, start=1):
        commit["rank_score"] = 1 / rank

    commits.sort(
        key=lambda x: abs(x.get("reserve_time_diff", float('inf')) if x.get("reserve_time_diff") is not None else float('inf'))
    )  # sort by Reserve Time Diff in ascending order
    for rank, commit in enumerate(commits, start=1):
        commit["reserve_time_diff_rank_score"] = 1 / rank

    commits.sort(
        key=lambda x: abs(x.get("publish_diff", float('inf')) if x.get("publish_diff") is not None else float('inf'))
    )  # sort by Publish Time Diff in ascending order
    for rank, commit in enumerate(commits, start=1):
        commit["publish_time_diff_rank_score"] = 1 / rank

    for commit in commits:
        commit["new_score"] = (
            bm25_weight * commit.get("rank_score", 0) +
            reserve_time_weight * commit.get("reserve_time_diff_rank_score", 0) +
            publish_time_weight * commit.get("publish_time_diff_rank_score", 0)
        )

    return commits

def process_file(file_name):
    input_file_path = os.path.join(input_dir, file_name)
    output_file_path = os.path.join(output_dir, file_name)

    with open(input_file_path, "r") as f:
        commits = json.load(f)

    commits = reciprocal_rank(commits)
    commits.sort(key=lambda x: x["new_score"], reverse=True)


    with open(output_file_path, "w") as f:
        json.dump(commits, f, indent=4)

    return file_name

def calculate_recall(cve_row):

    cve_id = cve_row["cve"]
    patch = cve_row["patch"]
    cve_file_path = os.path.join(output_dir, f"{cve_id}.json")

    if not os.path.exists(cve_file_path): 
        return None

    with open(cve_file_path, "r") as f:
        ranked_commits = json.load(f)

    ranked_commit_ids = [item["commit_id"] for item in ranked_commits]

    recall_at_k = {k: 0 for k in k_values}
    for k in k_values:
        if patch in ranked_commit_ids[:k]: 
            recall_at_k[k] = 1
    
    try:
        rank = ranked_commit_ids.index(patch) + 1
    except ValueError:
        rank = None
        
    return {
        "cve": cve_id,
        "patch": patch,
        "rank": rank,
        "recall_at_k": recall_at_k
    }


def generate_low_recall_csv(results):
    low_recall_records = []

    for result in results:
        if result is not None:
            # if result["recall_at_k"][100] <= 0.7:
            low_recall_records.append({
                "cve": result["cve"],
                "patch": result["patch"],
                "rank": result["rank"]
            })

    df = pd.DataFrame(low_recall_records)
    output_csv = "./low_recall_cves.csv"
    df.to_csv(output_csv, index=False)
    print(f"Low recall CVEs saved to {output_csv}")


def calculate_recall_per_repo(valid_list):

    repo_recall_results = {}

    repos = valid_list["repo"].unique()

    for repo in repos:
        repo_valid_list = valid_list[valid_list["repo"] == repo]

        recall_results = {k: 0 for k in k_values}
        total_cves = 0

        with Pool(cpu_count()) as pool:
            results = list(pool.imap(calculate_recall, repo_valid_list.to_dict("records")))

        for result in results:
            if result is not None:
                total_cves += 1
                for k in k_values:
                    recall_results[k] += result["recall_at_k"][k]

        try:
            recall_results = {k: recall_results[k] / total_cves for k in k_values}
        except ZeroDivisionError:
            recall_results = {k: 0 for k in k_values}

        repo_recall_results[repo] = recall_results

    return repo_recall_results

    
if __name__ == "__main__":
    files = os.listdir(input_dir)

    with Pool(cpu_count()) as pool:
        list(pool.imap(process_file, files))


    # valid_list = pd.read_csv(valid_list_path)
    # # valid_list = valid_list[valid_list["repo"] == "uaa"]

    # all_recalls = []
    # with Pool(cpu_count()) as pool:
    #     results = list(pool.imap(calculate_recall, valid_list.to_dict("records")))

    # recall_results = {k: 0 for k in k_values}
    # total_cves = 0

    # for result in results:
    #     if result is not None:
    #         total_cves += 1
    #         for k in k_values:
    #             recall_results[k] += result["recall_at_k"][k]

    # try:
    #     recall_results = {k: recall_results[k] / total_cves for k in k_values}
    # except ZeroDivisionError:
    #     import pdb; pdb.set_trace()

    # print("Recall@k Results:")
    # for k, recall in recall_results.items():
    #     print(f"Recall@{k}: {recall:.4f}")

    # output_path = f"./recall_results_{bm25_weight}_{reserve_time_weight}_{publish_time_weight}.json"
    # with open(output_path, "w") as f:
    #     json.dump(recall_results, f, indent=4)

    # generate_low_recall_csv(results)