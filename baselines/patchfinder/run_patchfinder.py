import json
import pathlib

import pandas as pd
from tqdm import tqdm
from transformers import set_seed

from bertscore import score_optimized
from tfidf import compute_tfidf_similarity
from data import load_json_file, get_commit_list, load_code_data


set_seed(42)

##################################################
# PATHS AND HYPERPARAMETERS
##################################################

N = 5000
owner_name = "xuxueli"
repo_name = "xxl-job"

LABEL_PATH = pathlib.Path("../../csv")
FEATURE_PATH = pathlib.Path("../../feature")
DATASETS_PATH = pathlib.Path("../../repo2commits_diff")

##################################################

label_df = pd.read_csv(LABEL_PATH / "AD.csv")
data = load_json_file(LABEL_PATH / pathlib.Path("cve2desc.json"))
cve_df = pd.DataFrame(list(data.items()), columns=['cve', 'desc'])
cve_df["desc_token"] = cve_df.desc.apply(lambda x: x[0]["value"])
cve_df = cve_df.drop(columns=["desc"])

label_df = pd.merge(left=label_df, right=cve_df, on="cve").rename(columns={"patch": "commit_id"})
label_df["name"] = label_df.apply(lambda x: "{}@@{}".format(x["owner"], x["repo"]), axis="columns")
label_df = label_df[label_df.name.isin([f"{owner_name}@@{repo_name}"])] # only include one library for demonstration purpose

group = label_df.groupby(by=["owner", "repo"]).first()
code_df = load_code_data(owner_name, repo_name, data_path=DATASETS_PATH)
partial_result_dfs = list()

for cve, ggroup in tqdm(group.groupby("cve"), total=group.cve.nunique()):
    # make sure there are n commit_ids per CVE, it may or may not contain the positive IDs
    sorted_commit_ids = get_commit_list(FEATURE_PATH, owner_name, repo_name, [cve], BM25_K=N)

    # len(positive_commit_ids) + len(negative_commit_ids) == n
    positive_commit_ids = [commit_id for commit_id in sorted_commit_ids if commit_id in ggroup.commit_id.tolist()]
    negative_commit_ids = [commit_id for commit_id in sorted_commit_ids if commit_id not in positive_commit_ids]

    sampled_code_df = pd.concat(
        [
            code_df[code_df.commit_id.isin(positive_commit_ids)],
            code_df[code_df.commit_id.isin(negative_commit_ids)]
        ]
    ).sample(frac=1)

    ##################################################
    # TF-IDF
    ##################################################
    partial_result_df = compute_tfidf_similarity(sampled_code_df, ggroup, verbose=False)

    ##################################################
    # CR SCORE
    ##################################################
    print("\nComputing CR Score...")

    p, r, f = score_optimized(
        cands=partial_result_df.desc_token.tolist(),
        refs=partial_result_df.commits.tolist(),
        model_type='microsoft/codereviewer',
        return_hash=False,
        batch_size=256,
        verbose=True,
        nthreads=20
    )
    partial_result_df = partial_result_df.assign(precison=p, recall=r, f1=f)

    ##################################################
    # FUSION
    ##################################################

    partial_result_df["score"] = partial_result_df.apply(lambda x: (x["similarity"] + x["f1"]) / 2, axis="columns")
    partial_result_dfs.append(partial_result_df)

df = pd.concat(partial_result_dfs).reset_index(drop=True)
df.to_pickle("results/patchfinder.pkl")

def save_result(df):
    SAVE_PATH = FEATURE_PATH / f"{owner_name}@@{repo_name}/patchfinder/result/"
    SAVE_PATH.mkdir(exist_ok=True, parents=True)

    for cve, group in df.groupby("cve"):
        result = group[["commit_id", "score"]].rename(columns={"score": "new_score"}).set_index("commit_id").to_dict()
        pathlib.Path(SAVE_PATH / f"{cve}.json").write_text(json.dumps(result, indent=4))

save_result(df)