import pandas as pd
import json
from pathlib import Path
import ast
import os
import difflib
from multiprocessing import Pool, cpu_count
import sys


def custom_path_similarity(path1, path2, base_weight=0.8, dir_weight=0.2):
    """
    Compute custom similarity between two file paths:
    - Use SequenceMatcher for the file name part (higher weight);
    - Use simple Jaccard similarity for the directory part (lower weight).
    """
    # Extract file name
    base1 = os.path.basename(path1)
    base2 = os.path.basename(path2)
    base_sim = difflib.SequenceMatcher(None, base1, base2).ratio()
    
    # Extract directory part
    dir1 = os.path.dirname(path1)
    dir2 = os.path.dirname(path2)
    tokens1 = set(dir1.split('/')) if dir1 else set()
    tokens2 = set(dir2.split('/')) if dir2 else set()
    if tokens1 or tokens2:
        dir_sim = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
    else:
        # If both have no directory information, treat directory similarity as 1.0
        dir_sim = 1.0

    # Combine the two parts
    score = base_weight * base_sim + dir_weight * dir_sim
    return score


dataset = sys.argv[1]
split = sys.argv[2]
# Read data
cve_df = pd.read_csv(f'../result/cve2name_with_paths_{dataset}_{split}.csv')
commit_df = pd.read_csv(f'../result/commits_file_path_{dataset}_{split}.csv')

# Build CVE -> file path set mapping, and also record repo info (assume column name is repo_key)
cve_path_dict = {}
cve_repo_mapping = {}
for idx, row in cve_df.iterrows():
    cve = row['cve']
    repo = row['repo_key']  # Assume this field exists in the CSV
    cve_repo_mapping[cve] = repo
    try:
        path_list = ast.literal_eval(row['path_in_repo'])
    except Exception:
        path_list = []
    cve_path_dict[cve] = {fp.strip() for fp in path_list}

# Build commit -> file path set mapping, and also record commit-to-repo mapping
commit_path_dict = {}
commit_repo_mapping = {}
for idx, row in commit_df.iterrows():
    commit = row['commit_id']
    repo = row['repo_key']
    commit_repo_mapping[commit] = repo
    try:
        path_list = ast.literal_eval(row['file_path'])
    except Exception:
        path_list = []
    commit_path_dict[commit] = {fp.strip() for fp in path_list}

# Pre-group commits by repo to reduce search space for each CVE
commits_by_repo = {}
for commit, repo in commit_repo_mapping.items():
    commits_by_repo.setdefault(repo, []).append((commit, commit_path_dict[commit]))

def process_single_cve(cve):
    """
    For a single CVE, compute similarity scores with all commits in the same repo,
    and write results to a JSON file.
    Use internal cache to avoid redundant computation.
    """
    cache = {}
    repo = cve_repo_mapping.get(cve)
    cve_paths = cve_path_dict[cve]
    result = {}
    # If no commits under this repo, return empty result
    for commit, commit_paths in commits_by_repo.get(repo, []):
        max_sim = 0.0
        for cve_path in cve_paths:
            for commit_path in commit_paths:
                key = (cve_path, commit_path)
                if key in cache:
                    sim = cache[key]
                else:
                    sim = custom_path_similarity(cve_path, commit_path)
                    cache[key] = sim
                if sim > max_sim:
                    max_sim = sim
        result[commit] = {"new_score": max_sim}
    # Write result to JSON file named {cve}.json
    output_file = f"../../feature/{repo}/jaccard/result/{cve}.json"
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
         json.dump(result, f, indent=4)
    return cve, result

if __name__ == '__main__':
    # Use all available CPU cores
    pool = Pool(cpu_count())
    # Process all CVEs in parallel, one CVE per worker
    cve_list = list(cve_path_dict.keys())
    results = pool.map(process_single_cve, cve_list)
    pool.close()
    pool.join()
    print("All CVEs processed!")
