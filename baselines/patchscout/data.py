import re
import os
import nltk
import json
import tqdm
import swifter
import pathlib
import multiprocessing

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from multiprocessing import Pool


stop_words = set(stopwords.words("english"))
LABEL_PATH = pathlib.Path("../../csv")

##################################################
# HELPERS
##################################################

def As_in_B(As: list, B:str):
    """
    obtain how many elements in A list appear in string B
    """
    cnt = 0
    for A in As:
        if A in B:
            cnt += 1
    return cnt


def file_match_func(filepaths, funcs, desc):
    files = [path.split("/")[-1] for path in filepaths]
    file_match = As_in_B(files, desc)
    filepath_match = As_in_B(filepaths, desc)
    func_match = As_in_B(funcs, desc)
    return file_match, filepath_match, func_match


def process_diff(diff):
    code_diff_tokens = []
    for line in diff.splitlines():
        if (line.startswith("+") and not line.startswith("++")) or (
            line.startswith("-") and not line.startswith("--")
        ):
            line = line.lower()
            tmp_list = []
            tmp_list = word_tokenize(line[1:])
            code_diff_tokens.extend(
                [token for token in tmp_list if token not in stop_words]
            )
    return code_diff_tokens


##################################################
# STEP 1: INITIAL PROCESSING
##################################################

##################################################
# CODE
##################################################

def re_filepath(item):
    res = []
    find = re.findall(
        '(([a-zA-Z0-9]|-|_|/)+\.(cpp|cc|cxx|cp|CC|hpp|hh|C|c|h|py|php|java))',
        item)
    for item in find:
        res.append(item[0])
    return res


def re_file(item):
    res = []
    find = re.findall(
        '(([a-zA-Z0-9]|-|_)+\.(cpp|cc|cxx|cp|CC|hpp|hh|C|c|h|py|php|java))',
        item)
    for item in find:
        res.append(item[0])
    return res


def re_func(item):
    res = []
    find = re.findall("(([a-zA-Z0-9]+_)+[a-zA-Z0-9]+.{2})", item)
    for item in find:
        item = item[0]
        if item[-1] == ' ' or item[-2] == ' ':
            res.append(item[:-2])

    find = re.findall("(([a-zA-Z0-9]+_)*[a-zA-Z0-9]+\(\))", item)
    for item in find:
        item = item[0]
        res.append(item[:-2])

    find = re.findall("(([a-zA-Z0-9]+_)*[a-zA-Z0-9]+ function)", item)
    for item in find:
        item = item[0]
        res.append(item[:-9])
    return res

def get_tokens(text, List):
    return set([item for item in List if item in text])

def get_code_info(cve, commit, diff):
    files, filepaths, funcs = [], [], []
    lines = diff.split('\n')
    for line in lines:
        if line.startswith('diff --git'):
            line = line.lower()
            files.append(line.split(' ')[-1].strip().split('/')[-1])
            filepaths.append(line.split(" ")[-1].strip())
        elif line.startswith('@@ '):
            line = line.lower()
            funcs.append(line.split('@@')[-1].strip())
    return [cve, commit, files, filepaths, funcs]


def mid_func(item):
    return get_code_info(*item)


def multi_process_code(cves, commits, diffs, poolnum=5):
    length = len(commits)
    with Pool(poolnum) as p:
        # ret = list(
        #     tqdm.tqdm(p.imap(mid_func, zip(cves, commits, diffs )), total=length, desc='get commits info'))
        ret = list(p.imap(mid_func, zip(cves, commits, diffs )))
        p.close()
        p.join()
    return ret

##### several cves share commits

def get_code_info1(commit, diff):
    files, filepaths, funcs = [], [], []
    lines = diff.split('\n')
    for line in lines:
        if line.startswith('diff --git'):
            line = line.lower()
            files.append(line.split(' ')[-1].strip().split('/')[-1])
            filepaths.append(line.split(" ")[-1].strip())
        elif line.startswith('@@ '):
            line = line.lower()
            funcs.append(line.split('@@')[-1].strip())
    return [commit, files, filepaths, funcs]


def mid_func1(item):
    return get_code_info1(*item)


def multi_process_code1(commits, diffs, poolnum=5):
    length = len(commits)
    with Pool(poolnum) as p:
        ret = list(
            tqdm.tqdm(p.imap(mid_func1, zip(commits, diffs )), total=length, desc='get commits info'))
        p.close()
        p.join()
    return ret

##################################################
# MESSAGES
##################################################

with open(LABEL_PATH / "vuln_type_impact.json", 'r') as f:
    vuln_type_impact = json.load(f)

vuln_type = set(vuln_type_impact.keys())
vuln_impact = set()

def re_bug(item):
    find = re.findall('bug.{0,3}([0-9]{2, 5})', item)
    return set(find)


def re_cve(item):
    return set(re.findall('(cve-[0-9]{4}-[0-9]{1,7})', item))


def process_msg1(msg, commit_id):
    type_set = set()
    for value in vuln_type:
        if value in msg:
            type_set.add(value)

    impact_set = set()
    for value in vuln_impact:
        if value in msg:
            impact_set.add(value)

    bugs = re_bug(msg)
    cves = re_cve(msg)
    return [commit_id, bugs, cves, type_set, impact_set]


def process_msg(cve_name, msg, commit_id):
    type_set = set()
    for value in vuln_type:
        if value in msg:
            type_set.add(value)

    impact_set = set()
    for value in vuln_impact:
        if value in msg:
            impact_set.add(value)
    bugs = re_bug(msg)
    cves = re_cve(msg)
    return [cve_name, commit_id, bugs, cves, type_set, impact_set]



##################################################
# STEP 2: ADDITIONAL PROCESSING TO GENERATE FEATURES
##################################################

def get_vuln_idf(bug, links, cve, cves):
    cve_match = 0
    for item in cves:
        if item in cve.lower():
            cve_match = 1
            break

    bug_match = 0
    for link in links:
        if "bug" in link or "Bug" in link:
            for item in bug:
                if item.lower() in link:
                    bug_match = 1
                    break

    return bug_match, cve_match


def get_vuln_loc(nvd_items, commit_items):
    same_cnt = 0
    for commit_item in commit_items:
        for nvd_item in nvd_items:
            if nvd_item in commit_item:
                same_cnt += 1
                break
    if len(commit_items) == 0:
        same_ratio = 0
    else:
        same_ratio = same_cnt / (len(commit_items))

    unrelated_cnt = len(nvd_items) - same_cnt
    return same_cnt, same_ratio, unrelated_cnt


def get_vuln_type_relevance(
    nvd_type, nvd_impact, commit_type, commit_impact, vuln_type_impact
):
    l1, l2, l3 = 0, 0, 0

    # Calculate l1
    if nvd_type and commit_type:
        for nvd_item in nvd_type:
            if nvd_item in commit_type:
                l1 += 1
    # l1 = len(nvd_type & commit_type)

    # Calculate l2 and l3
    if nvd_type and commit_impact:
        for nvd_item in nvd_type:
            impact_list = vuln_type_impact.get(nvd_item)
            if impact_list is None:
                l3 += 1
                continue
            for commit_item in commit_impact:
                if commit_item in impact_list:
                    l2 += 1
                else:
                    l3 += 1

    if commit_type and nvd_impact:
        for commit_item in commit_type:
            impact_list = vuln_type_impact.get(commit_item)
            if impact_list is None:
                l3 += 1
                continue
            for nvd_item in nvd_impact:
                if nvd_item in impact_list:
                    l2 += 1
                else:
                    l3 += 1

    cnt = l1 + l2 + l3 + 1
    return l1 / cnt, l2 / cnt, (l3 + 1) / cnt


def count_shared_words_dm(nvd_desc, commit_msg):
    # Tokenize the strings into words
    tokens1 = nltk.word_tokenize(nvd_desc.lower())
    tokens2 = nltk.word_tokenize(commit_msg.lower())

    # Remove stop words from the tokenized strings
    nvd_desc_tokens = [word for word in tokens1 if word not in stop_words]
    commit_msg_tokens = [word for word in tokens2 if word not in stop_words]

    # Get the shared words between the two tokenized strings
    shared_words = set(nvd_desc_tokens) & set(commit_msg_tokens)

    # Compute the frequency of each shared word in both tokenized strings
    nvd_desc_counts = {}
    commit_msg_counts = {}
    for word in shared_words:
        nvd_desc_counts[word] = nvd_desc_tokens.count(word)
        commit_msg_counts[word] = commit_msg_tokens.count(word)

    # Calculate the number of words in nvd description
    num_words_nvd = len(nvd_desc_tokens)

    # Calculate the Shared-Vul-Msg-Word Ratio
    svmw_ratio = len(shared_words) / (num_words_nvd + 1)

    # Calculate the maximum frequency of the shared words
    max_freq = max(
        list(nvd_desc_counts.values()) + list(commit_msg_counts.values()), default=0
    )

    # Calculate the sum of the frequencies of the shared words
    freq_sum = (
        sum(list(nvd_desc_counts.values())) + sum(list(commit_msg_counts.values()))
        if len(shared_words) > 0
        else 0
    )

    # Calculate the average frequency of the shared words
    freq_avg = (
        np.mean(list(nvd_desc_counts.values()) + list(commit_msg_counts.values()))
        if len(shared_words) > 0
        else 0
    )

    # Calculate the variance of the frequency of the shared words
    freq_var = (
        np.var(list(nvd_desc_counts.values()) + list(commit_msg_counts.values()))
        if len(shared_words) > 0
        else 0
    )

    # Return a tuple containing the number of shared words and the computed statistics
    return len(shared_words), svmw_ratio, max_freq, freq_sum, freq_avg, freq_var


def count_shared_words_dc(nvd_desc, code_diff):
    nvd_desc_tokens = word_tokenize(nvd_desc.lower())
    nvd_desc_tokens = [token for token in nvd_desc_tokens if token not in stop_words]

    code_diff_tokens = process_diff(code_diff)
    # code_diff_tokens = [token for token in code_diff_tokens if token not in stop_words]

    shared_words = set(nvd_desc_tokens) & set(code_diff_tokens)

    # Compute the frequency of each shared word in both tokenized strings
    nvd_desc_counts = {}
    code_diff_counts = {}
    for word in shared_words:
        nvd_desc_counts[word] = nvd_desc_tokens.count(word)
        code_diff_counts[word] = code_diff_tokens.count(word)

    # Calculate the number of words in nvd description
    num_words_nvd_desc = len(nvd_desc_tokens)

    # Calculate the Shared-Vul-Msg-Word Ratio
    svmw_ratio = len(shared_words) / (num_words_nvd_desc + 1)

    # Calculate the maximum frequency of the shared words
    max_freq = max(
        list(nvd_desc_counts.values()) + list(code_diff_counts.values()), default=0
    )

    # Calculate the sum of the frequencies of the shared words
    freq_sum = (
        sum(list(nvd_desc_counts.values())) + sum(list(code_diff_counts.values()))
        if len(shared_words) > 0
        else 0
    )

    # Calculate the average frequency of the shared words
    freq_avg = freq_sum / (len(nvd_desc_counts) + len(code_diff_counts) + 1)

    # Calculate the variance of the frequency of the shared words
    freq_var = (
        np.var(list(nvd_desc_counts.values()) + list(code_diff_counts.values()))
        if len(shared_words) > 0
        else 0
    )
    # Return the computed statistics
    return len(shared_words), svmw_ratio, max_freq, freq_sum, freq_avg, freq_var


def get_patchscout_features(df, verbose=False):
    # required input columns: "cve", "cvedesc", "commit_id", "diff", "commit_msg",
    # make assertations here to confirm these columns exists

    required_columns = {
        "cve", "cvedesc", "commit_id", "diff", "commit_msg"
    }
    assert required_columns.issubset(df.columns), f"Missing columns: {required_columns - set(df.columns)}"

    ##################################################

    df["links"] = df["commit_msg"].swifter.progress_bar(False).apply(
        lambda x: re.findall(r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', x)
    )

    ##################################################

    df['functions'] = df['cvedesc'].swifter.progress_bar(verbose).apply(re_func)
    df['files'] = df['cvedesc'].swifter.progress_bar(verbose).apply(re_file)
    df['filepaths'] = df['cvedesc'].swifter.progress_bar(verbose).apply(re_filepath)
    df['vuln_type'] = df['cvedesc'].swifter.progress_bar(verbose).apply(lambda item: get_tokens(item, vuln_type))
    df['vuln_impact'] = df['cvedesc'].swifter.progress_bar(verbose).apply(lambda item: get_tokens(item, vuln_impact))

    ##################################################

    if not df[df.label == 1].empty:
        df.loc[df.label == 1, ['code_files', 'code_filepaths', 'code_funcs']] = (
            df.loc[df.label == 1].swifter.progress_bar(verbose).apply(
                lambda x: pd.Series(get_code_info(x["cve"], x["commit_id"], x["diff"])[-3:]),
                axis="columns"
            )
        )

    if not df[df.label == 0].empty:
        df.loc[df.label == 0, ['code_files', 'code_filepaths', 'code_funcs']] = (
            df.loc[df.label == 0].swifter.progress_bar(verbose).apply(
                lambda x: pd.Series(get_code_info1(x["commit_id"], x["diff"])[-3:]),
                axis="columns"
            )
        )

    ##################################################

    if not df[df.label == 1].empty:
        df.loc[df.label == 1, ['msg_bugs', 'msg_cves', 'msg_type', 'msg_impact']] = (
            df.loc[df.label == 1].swifter.progress_bar(verbose).apply(
                lambda x: pd.Series(process_msg(x["cve"], x["commit_msg"], x["commit_id"])[-4:]),
                axis="columns"
            )
        )

    if not df[df.label == 0].empty:
        df.loc[df.label == 0, ['msg_bugs', 'msg_cves', 'msg_type', 'msg_impact']] = (
            df.loc[df.label == 0].swifter.progress_bar(verbose).apply(
                lambda x: pd.Series(process_msg1(x["commit_msg"], x["commit_id"])[-4:]),
                axis="columns"
            )
        )

    ##################################################
    # additional step: fillna

    df[['code_files', 'code_filepaths', 'code_funcs']] = df[
        ['code_files', 'code_filepaths', 'code_funcs']].swifter.progress_bar(verbose).applymap(
        lambda x: x if isinstance(x, list) and x else list()
    )

    df[['msg_bugs', 'msg_cves', 'msg_type', 'msg_impact']] = df[
        ['msg_bugs', 'msg_cves', 'msg_type', 'msg_impact']].swifter.progress_bar(verbose).applymap(
        lambda x: x if isinstance(x, set) and x else set()
    )

    ##################################################
    # FEATURE GENERATION
    ##################################################

    df[["cve_match", "bug_match"]] = df.swifter.progress_bar(verbose).apply(
        lambda row: pd.Series(get_vuln_idf(row["msg_bugs"], row["links"], row["cve"], row["msg_cves"])),
        axis="columns"
    )

    #### VL
    df[["func_same_cnt", "func_same_ratio", "func_unrelated_cnt"]] = df.swifter.progress_bar(verbose).apply(
        lambda row: pd.Series(get_vuln_loc(row["functions"], row["code_funcs"])),
        axis="columns"
    )

    ## but file &&&& filepaths
    df[["file_same_cnt", "file_same_ratio", "file_unrelated_cnt"]] = df.swifter.progress_bar(verbose).apply(
        lambda row: pd.Series(get_vuln_loc(row["files"], row["code_files"])),
        axis="columns"
    )

    df[["filepath_same_cnt", "filepath_same_ratio", "filepath_unrelated_cnt"]] = df.swifter.progress_bar(verbose).apply(
        lambda row: pd.Series(get_vuln_loc(row["filepaths"], row["code_filepaths"])),
        axis="columns"
    )

    #### VT
    df[["vuln_type_1", "vuln_type_2", "vuln_type_3"]] = df.swifter.progress_bar(verbose).apply(
        lambda row: pd.Series(get_vuln_type_relevance(
            row["vuln_type"],
            row["vuln_impact"],
            row["msg_type"],
            row["msg_impact"],
            vuln_type_impact,
        )),
        axis="columns"
    )
    df["patch_like"] = 0.5

    #### VDT
    df[["msg_shared_num", "msg_shared_ratio", "msg_max", "msg_sum", "msg_mean", "msg_var"]] = df.swifter.progress_bar(verbose).apply(
        lambda row: pd.Series(count_shared_words_dm(row["cvedesc"], row["commit_msg"])),
        axis="columns"
    )

    df[["code_shared_num", "code_shared_ratio", "code_max", "code_sum", "code_mean", "code_var"]] = df.swifter.progress_bar(verbose).apply(
        lambda row: pd.Series(count_shared_words_dc(row["cvedesc"], row["diff"])),
        axis="columns",
    )

    return df


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