import swifter
import pandas as pd

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_tfidf_similarity(code_df, label_df, verbose=False):
    """
    Compute the pairwise TF-IDF similarity between CVE description '{desc_token}' and the f'{msg_token} {diff_token}' (
    i.e., commits)

    Args:
     code_df: ['commit_id', 'msg_token', 'diff_token', 'commits'],
     label_df: ['cve', 'owner', 'repo', 'commit_id', 'desc_token']

    Returns:
     df: the df of size len(code_df) * len(label_df) with
     ["cve", "owner", "repo", "commit_id", "desc_token", "msg_token", "diff_token", "commits", "similarity"]
    """

    vectorizer = TfidfVectorizer()
    vectorizer.fit(tqdm(code_df["commits"].fillna("").tolist(), desc="Fitting TF-IDF", disable=not verbose))

    code_texts = tqdm(code_df["commits"].tolist(), desc="Transforming Code Description + Diff", disable=not verbose)
    desc_texts = tqdm(label_df["desc_token"].tolist(), desc="Transforming CVE Description", disable=not verbose)

    code_tfidf_matrix = vectorizer.transform(code_texts)
    desc_tfidf_matrix = vectorizer.transform(desc_texts)

    ##################################################

    code_tfidf_dict = dict(zip(code_df["commit_id"], code_tfidf_matrix))
    desc_tfidf_dict = dict(zip(label_df["cve"], desc_tfidf_matrix))

    def compute_similarity(row):
        commit_vector = code_tfidf_dict.get(row["commit_id"])
        desc_vector = desc_tfidf_dict.get(row["cve"])
        if commit_vector is not None and desc_vector is not None:
            return cosine_similarity(commit_vector, desc_vector)[0, 0]
        return 0.0

    df = pd.merge(left=label_df.assign(key=1), right=code_df.assign(key=1), on="key").drop(columns=["key"])
    df = df.drop(columns=["commit_id_x"]).rename(columns={"commit_id_y": "commit_id"})

    df["label"] = df.commit_id.apply(lambda x: 1 if x in label_df.commit_id.values else 0)
    df["similarity"] = df.swifter.progress_bar(False).apply(compute_similarity, axis=1)

    return df