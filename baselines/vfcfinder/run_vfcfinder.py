import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pathlib
import vfcfinder

import pandas as pd
from tqdm import tqdm
from transformers import set_seed
from utils.common import load_json_file
from utils.data import load_code_data

set_seed(42)

##################################################
# PATHS AND HYPERPARAMETERS
##################################################

owner_name = "xuxueli"
repo_name = "xxl-job"

LABEL_PATH = pathlib.Path("../../csv")
FEATURE_PATH = pathlib.Path("../../feature")
DATASETS_PATH = pathlib.Path("../../repo2commits_diff")

##################################################

label_df = pd.read_csv(LABEL_PATH / "AD.csv")

def initialize_data(df):
    df = df.drop_duplicates(subset=["cve", "patch"])
    df["name"] = df.apply(lambda x: "{}@@{}".format(x["owner"], x["repo"]), axis="columns")

    cve_data = load_json_file(LABEL_PATH / pathlib.Path("cve2desc.json"))
    cve_df = pd.DataFrame(list(cve_data.items()), columns=['cve', 'desc'])
    cve_df["cvedesc"] = cve_df.desc.apply(lambda x: x[0]["value"])
    cve_df = cve_df.drop(columns=["desc"])

    df = pd.merge(left=df, right=cve_df, on=["cve"])

    return df

label_df = pd.read_csv(LABEL_PATH / "AD.csv")
label_df = initialize_data(label_df)

test_df = label_df[label_df.name.isin([f"{owner_name}@@{repo_name}"])]
code_df = load_code_data(owner_name, repo_name, data_path=DATASETS_PATH)


##################################################
# VFCFinder operates on GHSA IDs, so we need to do an explicit mapping first
# the following CVEs are the CVEs for the repo xuxueli/xxl-job

mapping_dict = {
    "CVE-2020-23814": "GHSA-pqqj-299w-wf53",
    "CVE-2020-29204": "GHSA-wc73-w5r9-x9pc",
    "CVE-2022-36157": "GHSA-7qq9-9g2w-56f9",
    "CVE-2022-43183": "GHSA-83w4-x5w9-hf4h"
}

##################################################

CLONE_PATH = "./clones/"  # clone path can be any location
ADVISORY_PATH = "./advisories/"

all_dfs = list()

for cve, ghsa in mapping_dict.items():
    try:
        df1 = vfcfinder.vfc_ranker.rank(
            clone_path=CLONE_PATH,
            advisory_path=f"{ADVISORY_PATH}/{ghsa}.json",
            return_results=True
        )
        df1 = df1[["sha", "ranking_prob"]].rename(columns={"sha": "commit_id", "ranking_prob": "score"})
        df1["cve"] = cve

        df2 = pd.DataFrame(
            {
                "cve": cve,
                "commit_id": code_df[~code_df.commit_id.isin(df1.commit_id)].commit_id,
                "score": 0,
            }
        )
        all_dfs.append(pd.concat([df1, df2]).reset_index(drop=True))

    except Exception:
        all_dfs.append(code_df[["commit_id"]].assign(score=0).assign(cve=cve))
        print("VFCFinder Failed...")

result_df = pd.concat(all_dfs).reset_index(drop=True)
result_df.to_pickle("results/vfcfinder.pkl")

def save_result(df):
    SAVE_PATH = FEATURE_PATH / f"{owner_name}@@{repo_name}/vfcfinder/result/"
    SAVE_PATH.mkdir(exist_ok=True, parents=True)

    for cve, group in df.groupby("cve"):
        result = group[["commit_id", "score"]].rename(columns={"score": "new_score"}).set_index("commit_id").to_dict()
        pathlib.Path(SAVE_PATH / f"{cve}.json").write_text(json.dumps(result, indent=4))

save_result(result_df)