import re
import json
import tqdm
import pandas as pd
import os
from split_multiple_repos import truncate_record
from multiprocessing import Pool
import sys

commit_pattern = r"Commit: (.+?) Datetime: (.+?)\n([\s\S]*?)(?=\nCommit:|\Z)"
diff_pattern = r"Commit: (\w+) Datetime: (.+?)\n(.*?)(?:\n(?=diff --git)(diff --git [\s\S]*?))?(?=\nCommit:|\Z)"

def parse_commit_file(file_path, pattern):
    with open(file_path, "r", encoding="latin1") as file:
        content = file.read()
    matches = re.findall(pattern, content, re.DOTALL)
    data = {}
    for match in tqdm.tqdm(matches, desc="Parsing commits"):
        commit_hash, datetime, commit_message = match
        data[commit_hash.strip()] = {
            "commit_id": commit_hash.strip(),
            "datetime": datetime.strip(),
            "commit_msg": commit_message.strip(),
            "diff": "" 
        }
    return data

def parse_diff_file(file_path, pattern):
    with open(file_path, "r", encoding="latin1") as file:
        content = file.read()
    matches = re.findall(pattern, content, re.DOTALL)
    data = {}
    for match in tqdm.tqdm(matches, desc="Parsing diffs"):
        commit_hash, datetime, commit_msg, diff_content = match
        data[commit_hash.strip()] = {
            "commit_id": commit_hash.strip(),
            "datetime": datetime.strip(),
            "diff": truncate_record({"diff": diff_content.strip()}, 5 * 1024 * 1024)["diff"]
        }
    return data

def merge_data(commits, diffs):
    commit_ids = set(commits.keys())  # convert to set
    for commit_id, diff_data in diffs.items():
        if commit_id in commit_ids:
            commits[commit_id]["diff"] = diff_data["diff"]
        else:
            commits[commit_id] = diff_data 
    return list(commits.values())

def save_to_json(data, output_path):
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

def process_repo(owner_repo, base_path="../../"):
    """
    Processes the specified repository's Commit and Diff data and saves it as a JSON file.

    :param owner_repo: Repository identifier in the format "owner@@repo".
    :param base_path: Base path for locating input files.
    """
    commit_file = f"{base_path}/explain_patch/data/commits/commits_{owner_repo}.txt"
    diff_file = f"{base_path}/explain_patch/data/diff/diff_{owner_repo}.txt"
    output_file = f"./repo2commits_diff/{owner_repo}.json"
    
    if os.path.exists(output_file):
        # print(f"Skipping {owner_repo}, file already exists.")
        return
    
    if not os.path.exists(commit_file):
        print(f"Skipping {owner_repo}, commit file does not exist.")
        return


    print(f"Processing repository: {owner_repo}")

    try:
        commits_data = parse_commit_file(commit_file, commit_pattern)
        diffs_data = parse_diff_file(diff_file, diff_pattern)
        merged_data = merge_data(commits_data, diffs_data)
        save_to_json(merged_data, output_file)
        print(f"Parsed {len(merged_data)} commits and saved to {output_file}")
    except MemoryError:
        print(f"MemoryError while processing: {diff_file}")


if __name__ == "__main__":
    dataset = sys.argv[1]
    df = pd.read_csv(f'../csv/{dataset}.csv')
    
    # skip repos (large repo or few cve)
    # 37G diff_owncast@@owncast.txt, 27G diff_heartexlabs@@label-studio.txt, 26G diff_HumanSignal@@label-studio.txt, 27G diff_ag-grid@@ag-grid.txt, 46g diff_meshery@@meshery.txt, 14G diff_stanfordnlp@@CoreNLP.txt
    large_repos = ["ag-grid", "owncast", "label-studio", "CoreNLP", "meshery"] 

    df = df[~df['repo'].isin(large_repos)]
    
    repos = (df['owner'] + '@@' + df['repo']).unique()

    output_dir = "./repo2commits_diff"
    repos_to_process = [repo for repo in repos if not os.path.exists(f"{output_dir}/{repo}.json")]
    # repos_to_process = [repo for repo in repos if os.path.exists(f"{output_dir}/{repo}.json")] # overwrite
    print(f"Processing {len(repos_to_process)} repos...")
    
    from multiprocessing import Pool
    num_workers = 4


    with Pool(processes=num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(process_repo, repos_to_process), total=len(repos_to_process), desc="Processing repos"):
            pass
