import re
import os
import json
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import sys
from sklearn.metrics import ndcg_score
import argparse

max_rank = 100000

def qrel2recall(qrel, total_relevant=None):
    
    mrr_val = 1.0 / (1 + [x for x in range(len(qrel)) if qrel[x] == 1][0])

    recallat = [10, 100, 500, 1000, 2000, 5000, 10000]

    recall = {k: sum(qrel[:k]) / total_relevant if total_relevant else 0 for k in recallat}
    ndcg = {k: ndcg_score([sorted(qrel, reverse=True)], [qrel], k=k) for k in recall.keys()}

    ret = {
        "mrr": mrr_val}
    for k in recall.keys():
        ret[f"recall{k}"] = recall[k]
        ret[f"ndcg{k}"] = ndcg[k]
    return ret

def get_recall_from_sortedcommits(cve, ranked_commit_ids, patch):
    qrel = []
    ranked_commit_ids = ranked_commit_ids[:10000]

    for commit_id in ranked_commit_ids:
        if commit_id not in patch:
            qrel.append(0)
        else:
            qrel.append(1)
    if sum(qrel) == 0:
        return {
            "mrr": 0,
            "recall10": 0,
            "recall100": 0,
            "recall500": 0,
            "recall1000": 0,
            "recall2000": 0,
            "recall5000": 0,
            "recall10000": 0,
            "ndcg10": 0,
            "ndcg100": 0,
            "ndcg500": 0,
            "ndcg1000": 0,
            "ndcg2000": 0,
            "ndcg5000": 0,
            "ndcg10000": 0
        }
    return qrel2recall(qrel, len(patch))



def load_and_sort_commits(cve_file_path):
    if not os.path.exists(cve_file_path):
        return None

    with open(cve_file_path, "r") as f:
        ranked_commits = json.load(f)

    if isinstance(ranked_commits, list):
        if len(ranked_commits) > 0 and isinstance(ranked_commits[0], dict):
            # list of dicts
            ranked_commits = {
                item["commit_id"]: item.get("new_score", item.get("score", 0))
                for item in ranked_commits
            }
        elif len(ranked_commits) > 0 and isinstance(ranked_commits[0], list) and len(ranked_commits[0]) == 2:
            # list of lists
            ranked_commits = {item[1]: item[0] for item in ranked_commits}

    # sort commit_id
    if ranked_commits and isinstance(list(ranked_commits.values())[0], dict):
        sorted_commit_ids = sorted(
            ranked_commits.keys(),
            key=lambda x: ranked_commits[x].get("new_score", -1),
            reverse=True
        )
    else:
        sorted_commit_ids = sorted(ranked_commits.keys(), reverse=True)

    return sorted_commit_ids


def calculate_recall(cve_and_cve_row):
    cve_id = cve_and_cve_row[0]  # cve row ID
    patch = list(cve_and_cve_row[1]["patch"])
    repo = list(cve_and_cve_row[1]["repo"])[0]
    owner = list(cve_and_cve_row[1]["owner"])[0]
    model_name = list(cve_and_cve_row[1]["model_name"])[0]
    suffix = list(cve_and_cve_row[1]["suffix"])[0]

    ranked_commits_dir = "./feature/" + owner + "@@" + repo + "/" + model_name + f"/result{suffix}/"
    cve_file_path = os.path.join(ranked_commits_dir, f"{cve_id}.json")

    sorted_commit_ids = load_and_sort_commits(cve_file_path)
    if sorted_commit_ids is None:
        return None

    return get_recall_from_sortedcommits(cve_id, sorted_commit_ids, patch)


def calculate_recall_by_repo(valid_list):
    repo_recall_results = {}
    repo_cve_counts = {}
    k = 10

    repo_owner_map = valid_list.set_index("repo")["owner"].to_dict()  

    #filtered_repos = [
    #    repo for repo in repo_commit_counts
    #    if repo_commit_counts[repo] > 0 
    #]
    #filtered_valid_list = valid_list[valid_list["repo"].isin(filtered_repos)]
    #if filtered_valid_list.empty:
    #    return {}, 0

    groupby_df = list(valid_list.groupby("cve"))
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(calculate_recall, groupby_df)))

    for idx, result in enumerate(results):
        if result is not None:
            repo = list(groupby_df[idx][1]["repo"])[0]
            owner =  list(groupby_df[idx][1]["owner"])[0]
            if repo not in repo_recall_results:
                repo_recall_results[repo] = {} #"owner": owner}
                repo_cve_counts[repo] = 0
            
            repo_cve_counts[repo] += 1
            for k in result.keys():
                repo_recall_results[repo].setdefault(k, 0)
                repo_recall_results[repo][k] += result[k]
            # repo_recall_results[repo]["mrr"] += result["mrr"]

    for repo in repo_recall_results:
        for k in list(repo_recall_results[repo]):
            repo_recall_results[repo][k] /= repo_cve_counts[repo] 
    
    return repo_recall_results, repo_cve_counts

def count_commit(cve_file):
    repo_commit_counts_local = {}
    cve_file_path = os.path.join(ranked_commits_dir, cve_file)

    if not os.path.exists(cve_file_path) or not cve_file.endswith(".json"):
        return repo_commit_counts_local

    with open(cve_file_path, "r") as f:
        try:
            ranked_commits = json.load(f)
        except json.JSONDecodeError:
            return repo_commit_counts_local
    
    cve_id = cve_file.replace(".json", "")
    repo = cve_to_repo_map.get(cve_id, None)
    if repo is None:
        return repo_commit_counts_local
    
    commit_count = len(ranked_commits) if isinstance(ranked_commits, (dict, list)) else 0
    repo_commit_counts_local[repo] = commit_count
    
    return repo_commit_counts_local

def count_repo_commit_cve_counts():
    repo_commit_counts = Manager().dict()
    repo_cve_counts = valid_list["repo"].value_counts().to_dict()

    cve_files = os.listdir(ranked_commits_dir)

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(count_commit, cve_files), total=len(cve_files), desc="Counting commits per repo"))

    for result in results:
        for repo, count in result.items():
            # repo_commit_counts[repo] = repo_commit_counts.get(repo, 0) + count
            repo_commit_counts[repo] = count

    return repo_commit_counts, repo_cve_counts

def calculate_recall_with_df(groupby_list, ranked_commits_dir, model_name, suffix):
    # calculate recall based on the valid_list dataframe, assuming results/ already exist

     #[(x, model_name) for x in list(valid_list.groupby("cve"))]
    for each_cve, each_row in groupby_list:
        this_repo = list(each_row["repo"])[0]
        this_owner = list(each_row["owner"])[0]
        if not os.path.exists(ranked_commits_dir + this_owner + "@@" + this_repo + "/" + model_name + f"/result{suffix}/" + each_cve + ".json"):
            print(this_owner, this_repo, each_cve)
            continue
    print("groupby_list", len(groupby_list))
    with Pool(10) as pool:
        results = list(tqdm(pool.imap_unordered(calculate_recall, groupby_list), total=len(groupby_list), desc="Calculating Recall@k"))

    recall_results = {} #k: 0 for k in k_values}
    total_cves = 0

    # voyage_recall = json.load(open("voyage_recall.json", "r"))

    for result_idx in range(len(results)):
        result = results[result_idx]
        #print(groupby_list[result_idx][1]["cve"], result)
        total_cves += 1
        if result is not None:
            for k in result.keys():
            #for k in k_values:
                recall_results.setdefault(k, {})
                recall_results[k][total_cves - 1] = result[k]

    recall_results = {k: sum(recall_results[k].values()) / total_cves for k in recall_results.keys()}

    return recall_results

def calculate_tmp_valid_recall(valid_groupby, tmp_output_dir):
    total_cves = 0
    aggregated = {}

    for group in valid_groupby:
        cve_id = group[0]
        df = group[1]
        patch = list(df[df["label"] == 1]["commit_id"])
        cve_file_path = os.path.join(tmp_output_dir, f"{cve_id}.json")
        if not os.path.exists(cve_file_path):
            continue

        sorted_commit_ids = load_and_sort_commits(cve_file_path)
        if sorted_commit_ids is None:
            continue

        recall_metrics = get_recall_from_sortedcommits(cve_id, sorted_commit_ids, patch)
        if recall_metrics is not None:
            for k, v in recall_metrics.items():
                aggregated[k] = aggregated.get(k, 0) + v
            total_cves += 1

    if total_cves == 0:
        return {}

    recall_results = {k: v / total_cves for k, v in aggregated.items()}
    return recall_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ltr")
    parser.add_argument("--dataset_name", type=str, default="AD")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    suffix = args.suffix
    valid_list_path = f"./csv/{dataset_name}_test.csv"
    ranked_commits_dir = "./feature/" #"./GithubAD_ranked_commits_bm25_time"
    # ranked_commits_dir = "./GithubAD_ranked_commits_bm25_time"
    
    valid_list = pd.read_csv(valid_list_path)
    valid_list["suffix"] = suffix
    valid_list["model_name"] = model_name
    # valid_list = valid_list[valid_list['owner'] == 'torvalds']
    # valid_list = valid_list[valid_list['repo'] == 'tensorflow']
    cve_to_repo_map = valid_list.set_index("cve")["repo"].to_dict()

    groupby_list = list(valid_list.groupby("cve"))
    groupby_list = [x for x in groupby_list if (list(x[1]["owner"])[0], list(x[1]["repo"])[0]) in [("xuxueli", "xxl-job")]]

    recall_results = calculate_recall_with_df(groupby_list, ranked_commits_dir, model_name, suffix)
    print("Overall Recall@k Results:")
    first_ndcg = first_recall = True
    for k in sorted(recall_results.keys(), key = lambda x: (x[0], int(re.search("(\d+)", x).group(1)) if x != "mrr" else 0)):
        recall = recall_results[k]
        if first_recall and k.startswith("rec"):
            print("Recall:")
            first_recall = False
        if first_ndcg and k.startswith("ndcg"):
            print("ndcg:")
            first_ndcg = False
        print(f"{recall:.4f}")

    sys.exit(1)


    output_path = "recall_at_k_results.json"
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(output_path, "w") as f:
        json.dump({"recall_result/recall_at_k": recall_results}, f, indent=4)

    print(f"Results saved to {output_path}")
    
    ################################################################################################################
    repo_commit_counts, _ = count_repo_commit_cve_counts()
    recall_results_by_repo, repo_cve_counts = calculate_recall_by_repo(valid_list)
    
    # output_path = "recall_at_k_results_by_repo.json"
    # with open(output_path, "w") as f:
    #     json.dump(recall_results_by_repo, f, indent=4)
    
    # print(f"Results saved to {output_path}")
    output_csv_path = f"recall_result/recall_at_k_results_{model_name}_{dataset_name}.csv"
    recall_df = pd.DataFrame.from_dict(recall_results_by_repo, orient='index')
    recall_df.reset_index(inplace=True)
    recall_df.rename(columns={"index": "repo"}, inplace=True)
    recall_df["commit_count"] = recall_df["repo"].map(lambda x: repo_commit_counts.get(x, 0))
    recall_df["cve_count"] = recall_df["repo"].map(lambda x: repo_cve_counts.get(x, 0))
    
    repo_owner_map = valid_list.set_index("repo")["owner"].to_dict()
    recall_df["owner"] = recall_df["repo"].map(lambda x: repo_owner_map.get(x))
    
    recall_df.to_csv(output_csv_path, index=False)
    
    
    # print(f"Results saved to {output_csv_path}")
