import json
import multiprocessing
import os
import pathlib
import swifter

import pandas as pd
from tqdm import tqdm

def load_json_file(filename):
    with open(filename, "r") as fp:
        data = json.load(fp)

    return data


def get_commit_list_cve(feature_path, owner, repo, this_cve, BM25_K=10000):
    # file_path = feature_path + owner + "@@" + repo + "/bm25_time/result/" + this_cve + ".json"
    file_path = feature_path / f"{owner}@@{repo}/bm25_time/result/{this_cve}.json"

    if not os.path.exists(file_path):
        return set([])

    commit_list = []
    commit2entry = json.load(open(file_path))

    # sorted_commit2entry = sorted(commit2entry.items(), key=lambda x: x[1].get("new_score", -1), reverse=True)
    sorted_commit2entry = sorted(commit2entry, key=lambda x: x.get("new_score", -1), reverse=True)
    BM25_K = len(sorted_commit2entry) if BM25_K is None else BM25_K

    commit_list = [
        d["commit_id"] for d in sorted_commit2entry if "new_score" in d
    ][:min(BM25_K, len(sorted_commit2entry))]

    return commit_list


def worker(args):
    feature_path, owner, repo, each_cve, BM25_K = args
    return get_commit_list_cve(feature_path, owner, repo, each_cve, BM25_K)


def get_commit_list(feature_path, owner, repo, cve_list, BM25_K=10000):
    """

    :param feature_path:
    :param owner:
    :param repo:
    :param cve_list:
    :param BM25_K: the function will return the entire commit list when BM25_K is None
    :return:
    """
    commit_list = set([])

    args_list = [(feature_path, owner, repo, each_cve, BM25_K) for each_cve in cve_list]

    with multiprocessing.Pool(processes=10) as pool:
        results = list(pool.imap_unordered(worker, args_list))

    for this_commit_list in results:
        commit_list.update(this_commit_list)

    return list(commit_list)


def extract_changed_lines(diff_text, n=1000):
    changed_lines = [
        line for line in diff_text.splitlines()[:n] if
        (line.startswith("+") and not line.startswith("+++")) or
        (line.startswith("-") and not line.startswith("---"))
    ]

    return "\n".join(changed_lines)


def load_code_data(owner, repo, data_path=None, enable_full_diff=False):
    """

    :param owner:
    :param repo:
    :param enable_full_diff: set to True when we want to process the entire diff file
    :return: a dataframe of 5 columns: ['commit_id', 'datetime', 'msg_token', 'diff_token', 'commits']
    """
    name = f"{owner}@@{repo}"
    try:
        code_df = pd.read_json(data_path / f"{name}.json")
        # # special processing for the non-standard data structure
        # code_df = code_df.T.reset_index().rename(columns={"index": "commit_id"})
    except Exception:
        return None

    code_df = code_df.rename(columns={"commit_msg": "msg_token", "diff": "diff_token"})

    ##################################################
    # apply the setting in the paper: select the first 1000 lines in the diff and only keep +/- lines

    if not enable_full_diff:
        code_df["diff_token"] = code_df.diff_token.swifter.progress_bar(False).apply(lambda x: extract_changed_lines(x, n=1000))

    ##################################################

    code_df["commits"] = code_df.swifter.progress_bar(False).apply(lambda x: "{} {}".format(x["msg_token"], x["diff_token"]), axis="columns")
    code_df["commits"] = code_df["commits"].fillna("")

    return code_df

