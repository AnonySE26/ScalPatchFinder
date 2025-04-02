import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from vfcfinder.utils import osv_helper, git_helper


def prepare_vfcfinder_data(
        advisory_path,
        clone_path,
        repo_name,
        repo_owner,
):
    GHSA_ID = advisory_path
    CLONE_DIRECTORY = clone_path

    # dynamically set variables
    PARENT_PATH = f"{str(Path(__file__).resolve().parent.parent)}/"

    #####################################################################################
    # load/parse report
    with open(f"{PARENT_PATH}data/osv_schema.json", "r") as f:
        osv_schema = json.load(f)
        f.close()

    # parse the JSON
    parsed = osv_helper.parse_osv(
        osv_json_filename=f"{GHSA_ID}",
        osv_schema=osv_schema,
    )

    # set a clone path
    CLONE_PATH = f"{CLONE_DIRECTORY}{repo_owner}/{repo_name}/"

    #####################################################################################
    # clone repo
    # NOTE: the repo is not cloned if it is already downloaded
    print(f"\nCloning repository: {repo_owner}/{repo_name}")
    git_helper.clone_repo(
        repo_owner=repo_owner, repo_name=repo_name, clone_path=CLONE_DIRECTORY
    )

    #####################################################################################
    # find fixed/vulnerable version
    fix_tag = parsed[1].fixed.iloc[-1]

    #####################################################################################
    # get the prior and fixed tag of the local repo
    tags = git_helper.get_prior_tag(
        repo_owner=repo_owner,
        repo_name=repo_name,
        clone_path=CLONE_DIRECTORY,
        target_tag=fix_tag,
    )

    # set the vulnerable/fixed tags
    repo_vuln_tag = tags["prior_tag"]
    repo_fix_tag = tags["current_tag"]

    #####################################################################################
    # commits
    #####################################################################################

    commits = git_helper.get_commits_between_tags(
        prior_tag=repo_vuln_tag,
        current_tag=repo_fix_tag,
        temp_repo_path=CLONE_PATH,
    )

    #####################################################################################
    # commits_diff
    #####################################################################################

    diffs = [
        pd.DataFrame(git_helper.git_diff(clone_path=CLONE_PATH, commit_sha=sha))
        for sha in tqdm(commits["sha"], total=len(commits))
    ]
    commits_diff = pd.concat(diffs, ignore_index=True)

    return commits, commits_diff


