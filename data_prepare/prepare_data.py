import json
import requests
import shutil
from tqdm import tqdm
import os
import subprocess
import pandas as pd
from typing import Dict, Tuple, List
from dateutil import parser
import re
import pickle
from data_prepare.debug import *

if not os.path.exists('data/commits_cache'): 
    os.makedirs('data/commits_cache')

home_dir = os.path.expanduser("~")
github_token = json.load(open(home_dir + "/secret.json", "r"))["github"]
def can_clone_github_repo(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"Bearer {github_token}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("private") is False
        print("return status wrong")
        return False
    except requests.exceptions.RequestException:
        return False

def clone_or_pull_repo(owner, repo, commit_id=None, remove_repo: bool = False):
    repo_path = os.path.join("repo", f"{owner}@@{repo}")
    commits_output_path = os.path.join("data/commits", f"commits_{owner}@@{repo}.txt")
    # cur_commit_id_output_path = os.path.join("data/commit_msg", f"{owner}@@{repo}/{commit_id}.txt")
    if  os.path.exists(commits_output_path):
       print(f"Skipping {owner}/{repo} as it has already been processed.")
       return repo_path
    try:
        if not os.path.exists(repo_path):
            print(f"Cloning {repo} repository...")
            # subprocess.run(
            #     ["git", "clone", "--bare", "--filter=blob:none", f"https://github.com/{owner}/{repo}.git", repo_path],
            #     check=True
            # )
            subprocess.run(
                ["git", "clone", f"https://github.com/{owner}/{repo}.git", repo_path],
                check=True
            )
            os.chdir(repo_path)
            fout = open(f"../../data/commits/commits_{owner}@@{repo}.txt", "w")
            subprocess.run(["git", "log", "--pretty=format:Commit: %H Datetime: %ad\n %B", "--date-order", "--reverse"], stdout=fout, stderr=subprocess.PIPE, text = True)
            fout.close()
            fout = open(f"../../data/diff/diff_{owner}@@{repo}.txt", "w")
            subprocess.run(["git", "log", "-p", "--pretty=format:Commit: %H Datetime: %ad\n %s", "--date-order", "--reverse"], stdout=fout, stderr=subprocess.PIPE, text = True)
            fout.close()
            os.chdir("../../")  # move back to the root directory
        else:
            print(f"Fetching latest changes for {repo}...")
            subprocess.run(["git", "-C", repo_path, "fetch", "--all"], check=True)
            # pass  #  no need to fetch
        
        # if commit_id is not None:  # output commit msg
        #     if not os.path.exists(f'./data/commit_msg/{owner}@@{repo}'):
        #         os.makedirs(f'./data/commit_msg/{owner}@@{repo}')
        #     otuput_f = f"../../data/commit_msg/{owner}@@{repo}/{commit_id}.txt"
        #     os.chdir(repo_path)
        #     if not os.path.exists(otuput_f):
        #         fout = open(otuput_f, "w")
        #         subprocess.run(["git", "log", "-1", "--format=%B", commit_id], stdout=fout, stderr=subprocess.PIPE, text=True)
        #         fout.close()
        #     os.chdir("../../")
        if remove_repo:
            shutil.rmtree(repo_path)
        return repo_path
    
    except subprocess.CalledProcessError:
        return None


def safe_read_lines(file_path: str) -> List[str]:

    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file {file_path} with {encoding}: {str(e)}")
            continue
    
    print(f"Warning: Failed to read {file_path} with all encodings, returning empty list")
    return []


def parse_commit_time(owner: str, repo: str, reload: bool = False) -> Dict[str, Tuple[int, str]]:
    """
    Given a repository's commits file, returns a dictionary where the key is the commit_id,
    and the value is a tuple containing (commit_time, commit_msg).
    """
    commit_time_cache_file_path = os.path.join("data/commits_cache", f"{owner}@@{repo}.pkl")
    
    if not reload and os.path.exists(commit_time_cache_file_path):
        try:
            with open(commit_time_cache_file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Error loading cache file {commit_time_cache_file_path}: {str(e)}")
            

    commit_time_file_path = os.path.join("data/commits", f"commits_{owner}@@{repo}.txt")
    # with open(commit_time_file_path, 'r') as f:
    file_lines = safe_read_lines(commit_time_file_path)
    commit_time_data = {}
    current_commit = None
    commit_msg = []
    
    for line in file_lines:
        line = line.strip()
        if re.match(r'^commit [0-9a-f]+$', line):
            # Save previous commit data if exists
            if current_commit is not None:
                commit_time_data[current_commit] = (commit_timestamp, '\n'.join(commit_msg).strip())
            
            # Start new commit
            # import pdb; pdb.set_trace()
            current_commit = line.split(' ')[1]
            commit_msg = []
            commit_timestamp = None
            
        elif line.startswith('Date:'):
            # Parse date line
            date_str = ' '.join(line.split()[1:])
            try:
                commit_timestamp = int(parser.parse(date_str).timestamp())
            except Exception as e:
                print(f"Error parsing date {date_str}: {str(e)}")
                commit_timestamp = None
        elif line.startswith('Merge:'):
            pass
        elif not line.isspace() and not line.startswith('Author:'):
            # Add non-empty lines that aren't author to commit message
            commit_msg.append(line)
    
    # Add the last commit
    if current_commit is not None:
        commit_time_data[current_commit] = (commit_timestamp, '\n'.join(commit_msg))
    
    with open(commit_time_cache_file_path, 'wb') as f:
        pickle.dump(commit_time_data, f)
    return commit_time_data



if __name__ == "__main__":
    ad_df = pd.read_csv('../csv/AD.csv', header=0)
    patchfinder_df = pd.read_csv('../csv/patchfinder.csv', header=0)
    data_df = pd.concat([ad_df, patchfinder_df], ignore_index=True)
    
    from multiprocessing import Pool

    def process_group(group_data):
        group_df, owner, repo = group_data
        cur_idx = 0
        repo_key = f"{owner}@@{repo}"
        if owner == 'torvalds' and repo == 'linux':
            return  
        
        for idx, row in group_df.iterrows():
            cur_idx += 1
            patch = row['patch']
            commit_id = patch.split('/')[-1]
            # commit_file_path = f"./data/commits/commits_{repo_key}.txt"
            # if os.path.exists(commit_file_path):
            #     print(f"Skipping {repo_key}: Commit file already exists.")
            #     return
            
            if can_clone_github_repo(owner, repo):
                clone_or_pull_repo(owner, repo, commit_id=commit_id, remove_repo=True)
                break
            # else:
            #     clone_or_pull_repo(owner, repo, commit_id=commit_id)

    # Create groups data
    groups_data = []
    for (owner, repo), group_df in data_df.groupby(['owner', 'repo']):
        groups_data.append((group_df, owner, repo))

    # Use multiprocessing pool
    with Pool(processes=15) as pool:
        list(tqdm(pool.imap(process_group, groups_data), total=len(groups_data), desc="Cloning repositories"))
