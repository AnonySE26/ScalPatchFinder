import os
import sys
import json
import time
import ast
import pandas as pd
from tqdm import tqdm
import voyageai
from voyageai.error import RateLimitError
from multiprocessing import Pool


# Initialize Voyage
with open("../../secret.json", "r") as f:
    secret = json.load(f)
token1 = secret.get("voyage")
token2 = secret.get("voyage_2")
token3 = secret.get("voyage_3")
tokens = [token1, token2, token3]
# voyageai.api_key = token 


base_dir = "../feature"
batch_size = 500  
n_processes = 8   

dataset = sys.argv[1]
fold = sys.argv[2]

cve_file_train = f"./cve2name_with_paths_{dataset}_train.csv"
cve_file_test = f"./cve2name_with_paths_{dataset}_test.csv"
commit_file_train = f"./commits_file_path_{dataset}_train.csv"
commit_file_test = f"./commits_file_path_{dataset}_test.csv"


###################################################################
# func
###################################################################
def fix_corrupt_json(file_path):
    """
    Fix corrupted JSON file:
    1. Reverse scan the file to find the last `]`
    2. Read up to that position and manually add `}`
    3. Write back to the file
    """
    try:
        with open(file_path, "rb") as f:
            f.seek(0, 2) 
            pos = f.tell()
            last_bracket_pos = -1

            while pos > 0:
                pos -= 1
                f.seek(pos)
                char = f.read(1).decode(errors="ignore")

                if char == "]":
                    last_bracket_pos = pos 
                    break

        if last_bracket_pos != -1:
            with open(file_path, "rb") as f:
                content = f.read(last_bracket_pos + 1).decode(errors="ignore")

            content += "}" 

            temp_file_path = file_path + ".tmp"
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            os.rename(temp_file_path, file_path)
            print(f"Fix completed: {file_path}")
            return True
        else:
            print(f"Fix failed: ']' not found, file may be corrupted {file_path}")
            return False
    except Exception as e:
        print(f"Error occurred during fix: {e}")
        return False


def load_json_with_fix(file_path):

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f) 
    except json.JSONDecodeError:
        print(f"JSON parsing failed, attempting to fix {file_path}")
        if fix_corrupt_json(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            print(f"Unable to fix {file_path}")
            return None

########################################################
# 1. Read and merge data
########################################################


if fold == "train":
    cve_data_train = pd.read_csv(cve_file_train)
    commit_data_train = pd.read_csv(commit_file_train)
    cve_data, commit_data = cve_data_train, commit_data_train
    repo2commits_file = f"../feature/repo2commits_{dataset}_500.json"
    with open(repo2commits_file, "r", encoding="utf-8") as f:
        repo2commits_all = json.load(f)
    
if fold == "test":
    cve_data_test = pd.read_csv(cve_file_test)
    commit_data_test = pd.read_csv(commit_file_test)
    cve_data, commit_data = cve_data_test, commit_data_test
    cve2repo2commit = pd.read_csv(f"./{dataset}_test_commits.csv")
# else:
#     cve_data_train = pd.read_csv(cve_file_train)
#     commit_data_train = pd.read_csv(commit_file_train)
#     cve_data_test = pd.read_csv(cve_file_test)
#     commit_data_test = pd.read_csv(commit_file_test)
#     cve_data = pd.concat([cve_data_train, cve_data_test], ignore_index=True)
#     commit_data = pd.concat([commit_data_train, commit_data_test], ignore_index=True)

cve_data["path_in_repo"] = cve_data["path_in_repo"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and x.strip() else []
)
commit_data["file_path"] = commit_data["file_path"].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and x.strip() else []
)


cve_grouped = cve_data.groupby("repo_key")
commit_grouped = commit_data.groupby("repo_key")

unique_repos = sorted(list(
    set(cve_data["repo_key"].unique()) | set(commit_data["repo_key"].unique())
))

# unique_repos = ["apache@@camel"]
# unique_repos = unique_repos[99:]
########################################################
# 2. multi-process embedding
########################################################

def generate_embedding_with_retry(text_list, model="voyage-3", max_retries=5, token=None):
    """
    Call Voyage's embed() method on a batch of texts (list[str]). Automatically retry on rate limiting.
    Returns a list of embeddings corresponding to the provided text_list.
    """
    retries = 0
    delay = 5 
    while retries < max_retries:
        try:
            vo = voyageai.Client(api_key=token) if token else voyageai.Client()
            response = vo.embed(text_list, model=model, input_type="document")
            return response.embeddings 
        except RateLimitError:
            print(f"{token}: [RateLimitError] Retrying in {delay} seconds...")
            time.sleep(delay)
            retries += 1
            delay *= 2
        except Exception as e:

            print(f"[Error] {e}, retry in {delay} seconds...")
            time.sleep(delay)
            retries += 1
    print("[Error] Exceeded max retries, returning None for this batch.")
    return [None]*len(text_list)

def embed_batch(text_batch):
    """
    Function to be executed by each process: calls generate_embedding_with_retry on a batch of texts.
    Returns a tuple (text_batch, embedding_batch) for easier assembly later.
    """
    EMBED_DIM = 1024  # Please adjust to match the actual model output dimension.
    final_embeddings = []
    token_choice = tokens[hash("".join(text_batch)) % len(tokens)]
    for text in text_batch:
        if not text.strip():
            # If the text is empty, return a zero vector.
            final_embeddings.append([0.0] * EMBED_DIM)
        else:
            emb = generate_embedding_with_retry([text], model="voyage-3", token=token_choice)
            # If a valid result is returned, take the first embedding; otherwise, return a zero vector.
            if emb and emb[0] is not None:
                final_embeddings.append(emb[0])
            else:
                final_embeddings.append([0.0] * EMBED_DIM)
    return (text_batch, final_embeddings)

def chunk_list(lst, chunk_size):
    """Helper function: splits a list into chunks of size chunk_size and returns a generator."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]


########################################################
# 3. Process each repository
#    - Generate cve2path.json / commit2path.json
#    - Generate cve2embedding.json / commit2embedding.json
#    - For incomplete content, process in batches + multi-process embedding
########################################################

with Pool(processes=n_processes) as pool:
    for repo in unique_repos:
        # if repo in ["apache@@camel", "apache@@tomcat", "backstage@@backstage", "bytecodealliance@@wasmtime"]:
        #     continue
        print(f"==== Processing repo: {repo} ====")
        repo_folder = os.path.join(base_dir, repo, "path")
        os.makedirs(repo_folder, exist_ok=True)

        # =========== 3.1 cve2path.json ===============
        cve2path_file = os.path.join(repo_folder, "cve2path.json")
        cve_sub = cve_grouped.get_group(repo) if repo in cve_grouped.groups else pd.DataFrame()
        cve2path = {}
        if not cve_sub.empty:

            for _, row in cve_sub.iterrows():
                cve_id = row["cve"]
                paths = row["path_in_repo"] if isinstance(row["path_in_repo"], list) else []
                cve2path[cve_id] = paths

            # with open(cve2path_file, "w", encoding="utf-8") as f:
            #     json.dump(cve2path, f, ensure_ascii=False, indent=2)
            # print(f"Saved {len(cve2path)} CVEs to {cve2path_file}")
        else:
            print(f"Repo {repo} has no CVE data.")

        # =========== 3.2 commit2path.json ===============
        commit2path_file = os.path.join(repo_folder, "commit2path.json")
        if fold == "train":
            if repo in repo2commits_all:
                commit_list = repo2commits_all[repo]
                commit2path = {commit_id: [] for commit_id in commit_list}
                
                commit_sub = commit_grouped.get_group(repo) if repo in commit_grouped.groups else pd.DataFrame()
                if not commit_sub.empty:
                    for _, row in commit_sub.iterrows():
                        commit_id = row["commit_id"]
                        if commit_id in commit2path:
                            paths = row["file_path"]  
                            commit2path[commit_id] = paths
            else:
                commit2path = {}
                print(f"Repo {repo} not found in repo2commits file.")          
        else:
            commit_sub = commit_grouped.get_group(repo) if repo in commit_grouped.groups else pd.DataFrame()
            if not commit_sub.empty:
                if os.path.exists(commit2path_file):
                    with open(commit2path_file, "r", encoding="utf-8") as f:
                        commit2path = json.load(f)
                else:
                    commit2path = {}

                for _, row in commit_sub.iterrows():
                    commit_id = row["commit_id"]
                    paths = row["file_path"]  # list of str
                    commit2path[commit_id] = paths
                
                if len(commit2path) > 10000:
                    print(f"{repo} has more than 10000 commits, filtering using BM25 results.")
                    repo_clean = repo.replace('/', '@@')
                    commit_set = set()
                    for cve_id in cve2path.keys():
                        matching_rows = cve2repo2commit[(cve2repo2commit["cve"] == cve_id) & (cve2repo2commit["repo"] == repo_clean)]
                        # print(f"matching rows: {len(matching_rows)}")
                        commit_set.update(matching_rows["commit_id"].tolist())
                        # print("len of commits", len(commit_set))
                    commit2path = {cid: paths for cid, paths in commit2path.items() if cid in commit_set}

                with open(commit2path_file, "w", encoding="utf-8") as f:
                    json.dump(commit2path, f, ensure_ascii=False, indent=2)
            else:
                commit2path = {}
                print(f"Repo {repo} has no Commit data.")

        # =========== 3.3 cve2embedding.json ===============
        cve2embedding_file = os.path.join(repo_folder, "cve2embedding.json")
        # if os.path.exists(cve2embedding_file):
        #     with open(cve2embedding_file, "r", encoding="utf-8") as f:
        #         cve2embedding = json.load(f)
        # else:
        cve2embedding = {}

        cve_ids_to_process = [cid for cid in cve2path.keys() if cid not in cve2embedding]
        if cve_ids_to_process:
            print(f"{repo} - Need to embed {len(cve_ids_to_process)} CVEs...")
            cve_text_list = [" ".join(cve2path[cid]).strip() for cid in cve_ids_to_process]

            text_chunks = list(chunk_list(cve_text_list, batch_size))

            results = []
            for text_batch, emb_list in pool.imap(embed_batch, text_chunks):
                results.append((text_batch, emb_list))


            all_embs = []
            for (text_batch, emb_list) in results:
                all_embs.extend(emb_list)

            for cve_id, emb in zip(cve_ids_to_process, all_embs):
                cve2embedding[cve_id] = emb

            with open(cve2embedding_file, "w", encoding="utf-8") as f:
                json.dump(cve2embedding, f)
        else:
            print(f"{repo} - All CVE embeddings already exist or no CVEs at all.")

        # =========== 3.4 commit2embedding.json ===============
        commit2embedding_file = os.path.join(repo_folder, "commit2embedding.json")
        print(f"{repo} - Processing Commits...", flush=True)
        if os.path.exists(commit2embedding_file):
            with open(commit2embedding_file, "r", encoding="utf-8") as f:
                commit2embedding = load_json_with_fix(commit2embedding_file)
        else:
            commit2embedding = {}

        commit_ids_to_process = [cid for cid in commit2path.keys() if cid not in commit2embedding]
        if commit_ids_to_process:
            print(f"{repo} - Need to embed {len(commit_ids_to_process)} Commits...")
            commit_text_list = [" ".join(commit2path[cid]) for cid in commit_ids_to_process]

            text_chunks = list(chunk_list(commit_text_list, batch_size))
            results = []
            for text_batch, emb_list in tqdm(pool.imap(embed_batch, text_chunks), total=len(text_chunks)):
                results.append((text_batch, emb_list))

            all_embs = []
            for (text_batch, emb_list) in results:
                all_embs.extend(emb_list)

            for commit_id, emb in zip(commit_ids_to_process, all_embs):
                commit2embedding[commit_id] = emb

            with open(commit2embedding_file, "w", encoding="utf-8") as f:
                json.dump(commit2embedding, f)
        else:
            print(f"{repo} - All Commit embeddings already exist or no Commits at all.")

        print(f"==== Done for repo: {repo} ====\n")

print("All repos finished.")