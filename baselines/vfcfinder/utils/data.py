import swifter
import pandas as pd


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
