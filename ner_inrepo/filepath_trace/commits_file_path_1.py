import os
import json
import pandas as pd
import multiprocessing
import glob
import sys


def extract_file_paths(commit_diff):
    """
    Extract file paths from diff content (remove 'a/' prefix),
    only keep files containing '.' and avoid duplicates.
    """
    result = []
    seen = set()
    for line in commit_diff.splitlines():
        if line.startswith("diff --git"):
            parts = line.split(" ")
            if len(parts) >= 3:
                file_path = parts[2][2:]  # Remove "a/" prefix
                # Check if the filename contains a dot to avoid adding directories or files without extensions
                if file_path not in seen:
                    seen.add(file_path)
                    result.append(file_path)
    return result

def process_repo(repo_info):
    owner = repo_info['owner']
    repo = repo_info['repo']
    json_file_path = f"../repo2commits_diff/{owner}@@{repo}.json"
    
    if not os.path.exists(json_file_path):
        print(f"file {json_file_path} doesn't exist, skip {owner}@@{repo}")
        return

    print(f"processing {owner}@@{repo} ...")
    
    # Create folder for owner@@repo
    output_dir = os.path.join(base_output_folder, f"{owner}@@{repo}/path")
    os.makedirs(output_dir, exist_ok=True)

    output_file_path_csv = os.path.join(output_dir, f"commits_file_{owner}@@{repo}.csv")
    if os.path.exists(output_file_path_csv):
        print(f"{owner}@@{repo} already processed, skipping.")
        return
    
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    commit_file_paths = []
    for commit in data:
        commit_id = commit.get("commit_id")
        diff = commit.get("diff", "")
        if commit_id and diff:
            file_paths = extract_file_paths(diff)
            if file_paths:
                # One record per commit, file_path is a list
                commit_file_paths.append({
                    "commit_id": commit_id,
                    "file_path": file_paths
                })
    
    df = pd.DataFrame(commit_file_paths)
    # Convert list to JSON string to store in CSV
    df["file_path"] = df["file_path"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    df.to_csv(output_file_path_csv, index=False)


def filter_code_files(file_list):
    """
    Filter out non-code files, exclude files with certain extensions,
    and limit to a maximum of 5 files.
    """
    
    excluded_exts = {
        '.gitignore', '.md', '.json', '.toml', '.lock',
        '.txt', '.yml', '.yaml', '.ini', '.cfg', '.xml',
        '.log', '.csv', '.pdf', '.doc', '.docx', '.rtf',
        '.png', '.jpg', '.jpeg', '.gif', '.ico',
        '.zip', '.tar', '.gz', '.exe', '.dll', '.so'
    }
    filtered = []
    for f in file_list:
        base = os.path.basename(f)
        # Skip files without a dot
        if '.' not in base:
            continue
        # Use lowercase extension
        ext = os.path.splitext(f.lower())[1]
        if ext in excluded_exts:
            continue
        filtered.append(f)
    # Return at most k files
    return filtered[:5]


def merge_csv_files(output_csv_path, allowed_repo_keys):
    """
    Merge CSV files under base_output_folder listed in repo_list_path,
    and save the final merged CSV to output_csv_path.
    """
    # Search for all CSV files, path: base_output_folder/*/path/commits_file_*.csv
    pattern = os.path.join(base_output_folder, "*", "path", "commits_file_*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print("No CSV files found for merging.")
        return

    df_list = []
    for file in csv_files:
        # Use folder name to get repo_key
        repo_key = os.path.basename(os.path.dirname(os.path.dirname(file)))
        if repo_key in allowed_repo_keys:
            try:
                df_temp = pd.read_csv(file)
                df_temp["repo_key"] = repo_key
                df_list.append(df_temp)
            except Exception as e:
                print(f"Error reading file {file}: {e}")

    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        def process_file_paths(fp_str):
            try:
                paths = json.loads(fp_str)
            except Exception:
                paths = []
            filtered = filter_code_files(paths)
            return json.dumps(filtered, ensure_ascii=False)
        
        merged_df["file_path"] = merged_df["file_path"].apply(process_file_paths)
        merged_df.to_csv(output_csv_path, index=False)
        print(f"Merged CSV saved to {output_csv_path}")
    else:
        print("No valid CSV files to merge.")


if __name__ == '__main__':
    # python3 commits_file_path_1.py patchfinder train
    dataset = sys.argv[1]
    type = sys.argv[2]
    base_output_folder = "../../feature"

    repo_list_path = f"../csv/{dataset}_{type}.csv"
    df_repos = pd.read_csv(repo_list_path)
    unique_repos = df_repos[['owner', 'repo']].drop_duplicates()
    unique_repos = unique_repos[~((unique_repos["owner"] == "torvalds") & (unique_repos["repo"] == "linux"))]
    unique_repos["repo_key"] = unique_repos["owner"] + "@@" + unique_repos["repo"]
    allowed_repo_keys = set(unique_repos["repo_key"].tolist())

    #######################################################
    # # test repos
    # allowed_repos = {

    #     "mindsdb@@mindsdb",
    #     "spring-projects@@spring-framework",
    #     "answerdev@@answer",
    #     "cloudfoundry@@uaa",
    #     "kubernetes@@kubernetes",
    #     "OpenNMS@@opennms",
    #     "vantage6@@vantage6"
    # }
    # unique_repos["repo_key"] = unique_repos["owner"] + "@@" + unique_repos["repo"]
    # unique_repos = unique_repos[~unique_repos["repo_key"].isin(allowed_repos)]
    #######################################################
    repo_records = unique_repos.to_dict(orient="records")
    pool = multiprocessing.Pool(processes=4)
    pool.map(process_repo, repo_records)
    pool.close()
    pool.join()

    # merged_csv_path = f"./commits_file_path_{dataset}_{type}.csv"
    # merge_csv_files(merged_csv_path, allowed_repo_keys)
    