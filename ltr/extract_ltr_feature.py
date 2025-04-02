import pandas as pd
import os
import numpy as np
import math
import json
import tqdm
import sys
sys.path.append("../")
from embedding.utils import get_commit_list_cve_train, get_commit_list_cve_test
from recall import get_recall_from_sortedcommits
from pandas import read_csv

# combine scores using reciprocal rank

if __name__ == "__main__":
    feature_path = "../feature/"

    dataset_name = sys.argv[1]
    is_train = True if len(sys.argv) > 2 and sys.argv[2] == "train" else False
    fold = "train" if is_train else "test"

    repo2cve2negcommits = json.load(open(f"{feature_path}/repo2cve2negcommits_{dataset_name}_500_unsampled.json" if dataset_name == "patchfinder" else f"{feature_path}/repo2cve2negcommits_{dataset_name}_500.json", "r"))

    cve_data = read_csv(f"../csv/{dataset_name}_{fold}.csv")

    groupby_list = sorted(list(cve_data.groupby(["owner", "repo", "cve"])), key = lambda x:x[0])

    test_data = []

    for (owner, repo, cve), each_row in tqdm.tqdm(groupby_list):
        if (owner, repo) == ("stanfordnlp", "CoreNLP"): continue
        #if (owner, repo) == ("swagger-api","swagger-codegen"): continue
        # if (owner, repo) != ("phpmyadmin", "phpmyadmin"):
        #         continue
        patch_list = each_row["patch"].tolist()
        if is_train and cve not in repo2cve2negcommits.get(owner + "@@" + repo, {}):
            continue
        poscommits = list(each_row["patch"].tolist())
        commit2feat = {}

        if not os.path.exists(f"{feature_path}/{owner}@@{repo}/grit_instruct_512_file/result_head1/{cve}.json"): continue

        grit_commit2score_head1 = json.load(open(f"{feature_path}/{owner}@@{repo}/grit_instruct_512_file/result_head1/{cve}.json", "r"))

        grit_commit2score_head2 = json.load(open(f"{feature_path}/{owner}@@{repo}/grit_instruct_512_file/result_head2/{cve}.json", "r"))

        grit_commit2score = json.load(open(f"{feature_path}/{owner}@@{repo}/grit_instruct/result/{cve}.json", "r"))

        grit_commit2score_max = json.load(open(f"{feature_path}/{owner}@@{repo}/grit_instruct_512_file/result_head5_max/{cve}.json", "r"))

        bm25_commit2score = json.load(open(f"{feature_path}/{owner}@@{repo}/bm25_time/result/{cve}.json", "r"))

        if os.path.exists(f"{feature_path}/{owner}@@{repo}/path/result/{cve}.json"):
            path_commit2score = json.load(open(f"{feature_path}/{owner}@@{repo}/path/result/{cve}.json", "r"))
        else:
            path_commit2score = {}

        if os.path.exists(f"{feature_path}/{owner}@@{repo}/jaccard/result/{cve}.json"):
            jaccard_commit2score = json.load(open(f"{feature_path}/{owner}@@{repo}/jaccard/result/{cve}.json", "r"))
        else:
            jaccard_commit2score = {}

        allcommits = set(grit_commit2score.keys()).union(set(grit_commit2score_head1.keys())).union(set(grit_commit2score_head2.keys())).union(set(grit_commit2score_max.keys()))

        for each_commit in allcommits:
            this_entry = {}
            if each_commit in poscommits:
                this_entry.update({"cve": cve, "repo": owner + "@@" + repo, "label": 1, "commit_id": each_commit})
            else:
                this_entry.update({"cve": cve, "repo": owner + "@@" + repo, "label": 0, "commit_id": each_commit})

            this_entry["grit_head1"] = grit_commit2score_head1.get(each_commit, {}).get("new_score", 0)
            this_entry["grit_head2"] = grit_commit2score_head2.get(each_commit, {}).get("new_score", 0)
            this_entry["grit"] = grit_commit2score.get(each_commit, {}).get("new_score", 0)
            this_entry["grit_max"] = grit_commit2score_max.get(each_commit, {}).get("new_score", 0)
            try:
                this_entry["path"] = path_commit2score.get(each_commit, {}).get("new_score", 0)
            except AttributeError:
                this_entry["path"] = 0
            try:
                this_entry["jaccard"] = jaccard_commit2score.get(each_commit, {}).get("new_score", 0)
            except AttributeError:
                this_entry["jaccard"] = 0
            if each_commit not in bm25_commit2score:
                continue
            this_entry["bm25"] = bm25_commit2score[each_commit]["new_score"]
            this_entry["reserve_time_diff"] = 1.0 / (1.0 + abs(bm25_commit2score[each_commit]["reserve_time_diff"]))
            this_entry["publish_time_diff"] = 1.0 / (1.0 + abs(bm25_commit2score[each_commit]["publish_diff"]))
            test_data.append(this_entry)

    test_data = pd.DataFrame(test_data)

    test_data.to_csv(f"{feature_path}/final_feature/{dataset_name}/{dataset_name}_{fold}_feature.csv", index=False)

