import pandas
from pandas import read_csv
import math
import sys

def get_repo_list(patchfinder_data, x_range):
    patchfinder_repo = set([])
    for x in x_range:
        owner = patchfinder_data.iloc[x]["owner"]
        repo = patchfinder_data.iloc[x]["repo"]
        patchfinder_repo.add(owner + "@@" + repo)
    return patchfinder_repo


if __name__ == "__main__":
    data_name = sys.argv[1]
    if data_name == "AD":
        patchfinder_data = read_csv("../patchfinder_test.csv")
        patchfinder_repo = get_repo_list(patchfinder_data, range(len(patchfinder_data)))
    cve_data = read_csv(f"../{data_name}.csv")
    data = read_csv(f"../../../feature/repo_{data_name}.csv").drop_duplicates(subset= ["owner", "repo"])
    test_repo_ids = set([])
    train_repos = []
    large_commit = [x for x in range(len(data)) if data.iloc[x]["commit_count"] > 5000]
    small_commit = [x for x in range(len(data)) if data.iloc[x]["commit_count"] < 5000]
    large_commit_score = [(x, data.iloc[x]["mrr"] + 0.1 * data.iloc[x]["recall100"] + 0.1 * data.iloc[x]["recall500"] + 0.1 * data.iloc[x]["recall1000"]) for x in large_commit]
    sorted_large_commit_score = sorted(large_commit_score, key = lambda x:x[1])
    for (x, score) in sorted_large_commit_score:
        if score < 0.3:
            test_repo_ids.add(x)
    small_commit_score = [(x, (data.iloc[x]["mrr"], (data.iloc[x]["mrr"] + 0.1 * data.iloc[x]["recall100"] + 0.1 * data.iloc[x]["recall500"] + 0.1 * data.iloc[x]["recall1000"]) / math.log(100 + data.iloc[x]["commit_count"]))) for x in small_commit]
    sorted_small_commit_score = sorted(small_commit_score, key = lambda x:x[1][1])
    for (x, score) in sorted_small_commit_score:
        if score[1] < 0.05:
            test_repo_ids.add(x)
    # pick the repos in test_repos with larger #cves
    test_repo_name = get_repo_list(data, test_repo_ids)
    overlapped_reponame = patchfinder_repo.intersection(test_repo_name)
    sorted_idx = sorted(test_repo_ids, key = lambda x:data.iloc[x]["cve_count"], reverse=True)[:150]
    test_repos = []
    for x in range(len(data)):
        owner_repo = data.iloc[x]["owner"] + "@@" + data.iloc[x]["repo"]
        if (x in sorted_idx):
            if owner_repo not in overlapped_reponame:
               test_repos.append(data.iloc[x])
        else:
            train_repos.append(data.iloc[x])
    test_data = pandas.DataFrame(test_repos)
    train_data = pandas.DataFrame(train_repos)
    test_data.to_csv(f"../../../feature/repo_{data_name}_test.csv")
    train_data.to_csv(f"../../../feature/repo_{data_name}_train.csv")

    train_data_repo = [x for x in zip(train_data["owner"], train_data["repo"])]
    test_data_repo = [x for x in zip(test_data["owner"], test_data["repo"])]

    cve_train_data = []
    cve_test_data = []
    missed_repo = set([])
    for x in range(len(cve_data)):
        owner = cve_data.iloc[x]["owner"]
        repo = cve_data.iloc[x]["repo"]
        if (owner, repo) in train_data_repo:
            cve_train_data.append(cve_data.iloc[x])
        elif (owner, repo) in test_data_repo:
            cve_test_data.append(cve_data.iloc[x])
        else:
            missed_repo.add(owner + "@@" + repo)

    pandas.DataFrame(cve_train_data).to_csv(f"../{data_name}_train.csv")
    pandas.DataFrame(cve_test_data).to_csv(f"../{data_name}_test.csv")
           
