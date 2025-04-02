import json
import pathlib

import pandas as pd
from transformers import set_seed

from pipeline import (
    NUM_FEATURES,
    prepare_data,
    train_patchscout,
    test_patchscout,
)
from data import load_json_file

set_seed(42)
LABEL_PATH = pathlib.Path("../../csv")
FEATURE_PATH = pathlib.Path("../../feature")


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

train_df = label_df[label_df.name.isin(["xCss@@Valine"])]
test_df = label_df[label_df.name.isin(["xuxueli@@xxl-job"])]

prepare_data(train_df, n=1000, split="train")
prepare_data(test_df, n=5000, split="test")

processed_train_df = pd.concat(
    [
        pd.read_pickle(filename) for filename in pathlib.Path("train").glob("*.pkl")
    ]
).reset_index(drop=True)

processed_test_df = pd.read_pickle("test/xuxueli@@xxl-job.pkl")

checkpoint_name = "patchscout.pth"
train_patchscout(processed_train_df, NUM_FEATURES, checkpoint_name, batch_size=1024)
pred_df = test_patchscout(processed_test_df, NUM_FEATURES, checkpoint_name, batch_size=8192)

pred_df.to_pickle("results/patchscout.pkl")


def save_result(df):
    SAVE_PATH = FEATURE_PATH / f"xuxueli@@xxl-job/patchscout/result/"
    SAVE_PATH.mkdir(exist_ok=True, parents=True)

    for cve, group in df.groupby("cve"):
        result = group[["commit_id", "score"]].rename(columns={"score": "new_score"}).set_index("commit_id").to_dict()
        pathlib.Path(SAVE_PATH / f"{cve}.json").write_text(json.dumps(result, indent=4))

save_result(pred_df)