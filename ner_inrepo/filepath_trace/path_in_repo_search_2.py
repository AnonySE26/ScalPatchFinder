import os
import pandas as pd
import json
import re
import requests
from tqdm import tqdm
import time
import multiprocessing
import sys
import argparse

def load_repo_mapping(dataset="AD"):

    test_csv_path = f"../csv/{dataset}_test.csv"
    train_csv_path = f"../csv/{dataset}_train.csv"
    dfs = []
    
    for csv_path in [test_csv_path, train_csv_path]:
        if os.path.exists(csv_path):
            try:
                df_csv = pd.read_csv(csv_path)
                df_csv = df_csv[df_csv["owner"]!= 'torvalds']
                if set(['cve', 'owner', 'repo']).issubset(df_csv.columns):
                    dfs.append(df_csv[['cve', 'owner', 'repo']])
                else:
                    print(f"{csv_path} is missing required fields")
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        else:
            print(f"{csv_path} does not exist.")
    
    if not dfs:
        print("Failed to load any owner/repo mapping data")
        return {}
    
    mapping_df = pd.concat(dfs).drop_duplicates(subset=["cve"])
    mapping_df.set_index("cve", inplace=True)
    mapping = mapping_df.to_dict(orient="index")
    return mapping



def search_paths(word, repo, token=None):
    """
    Search for the specified word in the GitHub repository file paths:
    - If the word looks like a filename, use the Git Trees API to get the full file tree and filter matches;
    - Otherwise, use Code Search to search for the keyword in file contents.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    # If it looks like a filename, use the Git Trees API
    if re.search(r"\.\w+$", word):
        # Get repository info to determine default branch
        repo_url = f"https://api.github.com/repos/{repo}"
        repo_resp = requests.get(repo_url, headers=headers)
        if repo_resp.status_code != 200:
            raise Exception(f"Failed to get repo info, status code: {repo_resp.status_code}, response: {repo_resp.text}")
        repo_info = repo_resp.json()
        default_branch = repo_info.get("default_branch", "master")
        
        # Get the full tree of the repository (recursive mode)
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{default_branch}?recursive=1"
        tree_resp = requests.get(tree_url, headers=headers)
        if tree_resp.status_code != 200:
            raise Exception(f"Failed to get repo tree, status code: {tree_resp.status_code}, response: {tree_resp.text}")
        tree_data = tree_resp.json().get("tree", [])
        
        # Filter paths that match the filename (e.g., exact match at the end)
        matching_paths = [item["path"] for item in tree_data
                          if item["type"] == "blob" and item["path"].endswith(word)]
        return matching_paths
    else:
        # Otherwise, use Code Search to search file contents
        url = f"https://api.github.com/search/code?q={word}+in:file+repo:{repo}"
        response = requests.get(url, headers=headers)
        if response.status_code == 403 and "rate limit exceeded" in response.text.lower():
            raise Exception("Rate limit exceeded")
        elif response.status_code != 200:
            raise Exception(f"GitHub API request failed, status code: {response.status_code}, response: {response.text}")
        data = response.json()
        paths = [item["path"] for item in data.get("items", [])]
        return paths


def process_row(cve, words, full_repo, token):
    """
    For a single search word, look up the owner and repo from the mapping using the CVE,
    then search in the specified repository, returning search results (only top 5).
    """
    repo_results = {}
    if pd.isna(words) or words == "None":
        return repo_results
    
    search_terms = [term.strip() for term in words.split(",") if term.strip()]
    
    # Search each term and merge the results
    for term in search_terms:
        while True:  # Retry indefinitely, wait on rate limit
            try:
                result = search_paths(term, repo=full_repo, token=token)
                if result:
                    # Keep only the top 5
                    if full_repo not in repo_results:
                        repo_results[full_repo] = []
                    repo_results[full_repo].extend(result[:5])
                break
            except Exception as e:
                if "Rate limit exceeded" in str(e):
                    # print(f"Rate limit exceeded for '{term}' in {full_repo}. Sleeping for 30 seconds...")
                    time.sleep(60)
                else:
                    print(f"Error searching for '{term}' in {full_repo}: {e}")
                    break
        time.sleep(1)
    print(f"{cve} {repo_results}", flush=True)
    return repo_results

def main():
    # python path_in_repo_search_4.py --dataset patchfinder --mode multi --processes 4
    # python path_in_repo_search_4.py --dataset patchfinder --mode single
    parser = argparse.ArgumentParser(description="GitHub file path search tool")
    parser.add_argument('--dataset', type=str, default="AD", help="Dataset name (e.g., AD or patchfinder)")
    parser.add_argument('--mode', type=str, default="multi", choices=["multi", "single"], help="multi for multiprocessing, single for single process")
    parser.add_argument('--processes', type=int, default=4, help="Number of processes to use in multiprocessing mode")
    parser.add_argument('--split', type=str, default="train", choices=["train", "test"], help="Choose dataset split: train or test")
    args = parser.parse_args()
    
    dataset = args.dataset
    mode = args.mode
    
    test_path = f"../ner/{dataset}_test_path.csv"
    train_path = f"../ner/{dataset}_train_path.csv"
    
    
    try:
        df_test = pd.read_csv(test_path)
        df_train = pd.read_csv(train_path)
    except Exception as e:
        print(f"Failed to read CSV files: {e}")
        sys.exit(1)
        
    # df = pd.concat([df_test, df_train])
    if args.split == "train":
        output_csv = f"./cve2name_with_paths_{dataset}_train.csv"
        df = df_train
    else:
        output_csv = f"./cve2name_with_paths_{dataset}_test.csv"
        df = df_test
    # Load owner/repo mapping, keyed by CVE
    mapping = load_repo_mapping(dataset=dataset)
    print(f"Loaded {len(mapping)} CVEs")
    
    # Read GitHub tokens
    try:
        with open("../../secret.json", "r") as f:
            secret = json.load(f)
    except Exception as e:
        print(f"Failed to read token file: {e}")
        sys.exit(1)
    token1 = secret.get("github_1", "")
    token2 = secret.get("github_2", "")
    token3 = secret.get("github_3", "")
    token4 = secret.get("github_4", "")
    token5 = secret.get("github_5", "")

    # Construct the list of tasks to process, each task corresponds to a row in the CSV
    # Use the mapping to find the corresponding owner and repo for each CVE, and construct full_repo
    tasks = []
    for idx, row in df.iterrows():
        cve = row["cve"]
        word = row["file_func_name"]
        if cve not in mapping:
            # print(f"No owner/repo information found for {cve}")
            continue
        owner = mapping[cve].get("owner")
        repo = mapping[cve].get("repo")
        if not owner or not repo:
            print(f"Incomplete owner or repo information for {cve}")
            continue
        full_repo = f"{owner.strip()}/{repo.strip()}"
        if idx % 5 == 0:
            token = token1
        elif idx % 5 == 1:
            token = token2
        elif idx % 5 == 2:
            token = token3
        elif idx % 5 == 3:
            token = token4
        else:
            token = token5
        tasks.append((cve, word, full_repo, token))
    
    if mode == "multi":
        print("Using multi-process mode")
        pool = multiprocessing.Pool(processes=args.processes)
        results = list(tqdm(pool.starmap(process_row, tasks), total=len(tasks), desc="Processing rows"))
        pool.close()
        pool.join()
    else:
        print("Using single-process mode")
        results = []
        for task in tqdm(tasks, total=len(tasks), desc="Processing rows"):
            # import pdb; pdb.set_trace()
            results.append(process_row(*task))
    
    new_repo_keys = []
    new_path_in_repo = []
    # Note: tasks and results are in one-to-one correspondence
    for res in results:
        if res:
            # Here res should have only one key (i.e., full_repo), extract it directly
            repo_key, paths = list(res.items())[0]
            new_repo_keys.append(repo_key)
            new_path_in_repo.append(paths)
        else:
            new_repo_keys.append("")
            new_path_in_repo.append([])

    # Filter rows where CVE exists in the mapping (corresponding to tasks)
    df = df.loc[df["cve"].isin(mapping.keys())].copy()
    df["repo_key"] = [key.replace("/", "@@") if isinstance(key, str) else key for key in new_repo_keys]
    df["path_in_repo"] = new_path_in_repo

    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to: {output_csv}")
    
    

if __name__ == '__main__':
    main()