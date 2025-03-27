import os
import random
import json
import tqdm
import pandas as pd
from pandas import read_csv
import sys
import math

random.seed(42)

if __name__ == "__main__":

    dataset_name = sys.argv[1]

    

    #feat2weight = {"grit_instruct": 1.0, "bm25": 0.2, "reserve": 0.2, "publish": 0.2, "path": 0.0}
    repo2cve2negcommits = {}
    repo2commits = {}
    cve2randomcommits = {}
    K = 500

    if dataset_name == "AD":

        this_K = 250
        data = read_csv(f"../csv/{dataset_name}_train_full.csv")
        # get 250 hard repos and 250 random repos
        repo_github = read_csv(f"../feature/repo_{dataset_name}_train.csv")
        hard_train = set([])
        large_commit = [x for x in range(len(repo_github)) if repo_github.iloc[x]["commit_count"] > 5000]
        small_commit = [x for x in range(len(repo_github)) if repo_github.iloc[x]["commit_count"] < 5000]
        large_commit_score = [(x, repo_github.iloc[x]["mrr"] + 0.1 * repo_github.iloc[x]["recall100"] + 0.1 * repo_github.iloc[x]["recall500"] + 0.1 * repo_github.iloc[x]["recall1000"]) for x in large_commit]
        sorted_large_commit_score = sorted(large_commit_score, key = lambda x:x[1])
        for (x, score) in sorted_large_commit_score:
            if score < 0.4:
                hard_train.add(x)
        small_commit_score = [(x, (repo_github.iloc[x]["mrr"], (repo_github.iloc[x]["mrr"] + 0.1 * repo_github.iloc[x]["recall100"] + 0.1 * repo_github.iloc[x]["recall500"] + 0.1 * repo_github.iloc[x]["recall1000"]) / math.log(100 + repo_github.iloc[x]["commit_count"]))) for x in small_commit]
        sorted_small_commit_score = sorted(small_commit_score, key = lambda x:x[1][1])
        for (x, score) in sorted_small_commit_score:
            if score[1] < 0.06:
                hard_train.add(x)
        sorted_idx = sorted(hard_train, key = lambda x:repo_github.iloc[x]["cve_count"], reverse=True)[:this_K]
        random_idx = random.sample(range(len(repo_github)), this_K)
        
        hard_train_repos = set([])

        for x in range(len(repo_github)):
            #owner_repo = repo_github.iloc[x]["owner"] + "@@" + repo_github.iloc[x]["repo"]
            if (x in sorted_idx) or (x in random_idx):
                hard_train_repos.add(repo_github.iloc[x]["owner"] + "@@" + repo_github.iloc[x]["repo"])

        data = data[data.apply(lambda x: x["owner"] + "@@" + x["repo"] in hard_train_repos, axis=1)]
        data.to_csv(f"../csv/{dataset_name}_train.csv")
        
        # json1 = json.load(open(f"../feature/repo2cve2negcommits_{dataset_name}_500_full.json", "r"))
        # json2 = json.load(open(f"../feature/repo2commits_{dataset_name}_500_full.json", "r"))
        # new_json1 = {key: val for key, val in json1.items() if key in hard_train_repos}
        # new_json2 = {key: val for key, val in json2.items() if key in hard_train_repos}
        # json.dump(new_json1, open(f"../feature/repo2cve2negcommits_{dataset_name}_500.json", "w"))
        # json.dump(new_json2, open(f"../feature/repo2commits_{dataset_name}_500.json", "w"))
        
    else:
        data = read_csv(f"../csv/{dataset_name}_train.csv")
    groupby_list = list(data.groupby(["owner", "repo", "cve"]))


    for (owner, repo, cve), each_row in tqdm.tqdm(groupby_list):
        if (owner, repo) != ("xCss", "Valine"): continue
        patch = list(each_row["patch"].tolist())
        each_file = owner + "@@" + repo
        # get bm25 sorted list for the cve
        path = f"../feature/{owner}@@{repo}/bm25_time/result/"
        if not os.path.exists(path + cve + ".json"): continue

        repo2commits.setdefault(each_file, set([]))
        repo2commits[each_file].update(patch)

        commit2score = json.load(open(path + cve + ".json", "r"))
        commit2bm25 = {commit: commit2score[commit]["new_score"] for commit in commit2score}

        sorted_commit2bm25 = sorted(commit2bm25.items(), key = lambda x:x[1], reverse=True)
        sorted_commits = [sorted_commit2bm25[x][0] for x in range(len(sorted_commit2bm25))] 

        print("size", owner, repo, cve, len(sorted_commits))

        repo2cve2negcommits.setdefault(each_file, {})
        repo2cve2negcommits[each_file].setdefault(cve, {})

        random_commits = cve2randomcommits.get(cve, [])
        if len(random_commits) == 0:
            random_commits = random.sample(sorted_commits, min(K, len(sorted_commits)))
            cve2randomcommits[cve] = random_commits

        repo2cve2negcommits[each_file][cve]["random"] = list(cve2randomcommits[cve])
        repo2commits[each_file].update(cve2randomcommits[cve])

        top_commits = sorted_commits[:K]
        repo2cve2negcommits[each_file][cve]["top"] = list(top_commits)
        repo2commits[each_file].update(top_commits)

    # for each_file in list(repo2cve2negcommits.keys()):
    #     for cve in list(repo2cve2negcommits[each_file].keys()):
    #         repo2cve2negcommits[each_file][cve] = list(repo2cve2negcommits[each_file][cve])
    json.dump(repo2cve2negcommits, open(f"../feature/repo2cve2negcommits_{dataset_name}_{K}.json", "w"))
    for each_file in list(repo2commits.keys()):
        repo2commits[each_file] = list(repo2commits[each_file])
    json.dump(repo2commits, open(f"../feature/repo2commits_{dataset_name}_{K}.json", "w"))

