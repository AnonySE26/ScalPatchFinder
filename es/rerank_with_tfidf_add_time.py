import os
import json
import sys
from multiprocessing import Pool, cpu_count
import pandas as pd

# python rerank_with_tfidf_add_time.py 0.5 0.3 0.2 AD
if len(sys.argv) != 5:
    print("Usage: python rerank_with_tfidf_add_time.py <bm25_weight> <reserve_time_weight> <publish_time_weight> <dataset>")
    sys.exit(1)

bm25_weight = float(sys.argv[1])
reserve_time_weight = float(sys.argv[2])
publish_time_weight = float(sys.argv[3])
dataset = sys.argv[4]
input_dir = f'./{dataset}_ranked_commits_bm25_time'

valid_list_path = f"../csv/{dataset}.csv"
df = pd.read_csv(valid_list_path)
df = df[~((df['owner'] == 'torvalds') & (df['repo'] == 'linux'))]
cve_to_repo = {row["cve"]: f"{row['owner']}@@{row['repo']}" for _, row in df.iterrows()}

# k_values = [1, 10, 100, 500, 1000, 2000, 3000, 4000, 5000]
k_values = [100, 500, 1000, 2000, 5000]

def get_output_directory_for_cve(cve):
    repo_folder = cve_to_repo.get(cve)
    if repo_folder is None:
        return None
    output_dir = os.path.join("../feature", repo_folder, "bm25_time", "result")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


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
    cve = os.path.splitext(file_name)[0]
    output_dir = get_output_directory_for_cve(cve)
    output_file_path = os.path.join(output_dir, file_name)

    with open(input_file_path, "r") as f:
        commits = json.load(f)

    commits = reciprocal_rank(commits)
    commits.sort(key=lambda x: x["new_score"], reverse=True)


    with open(output_file_path, "w") as f:
        commits_dict = {commit["commit_id"]: commit for commit in commits}
        json.dump(commits_dict, f, indent=4)

    return file_name


    
if __name__ == "__main__":
    files = os.listdir(input_dir)

    with Pool(cpu_count()) as pool:
        list(pool.imap(process_file, files))
