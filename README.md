# SPFinder: A Scalable and Effective System for Tracing Known Vulnerability Patches

This repository contains the code and data sample for our submission: ScalPatchFinder: A Scalable and Effective Retrieval System for Security Vulnerability Patches.

<p align="center">
    <img src="figs/overview2.png"  width=800>
    <br>
</p>



## Overview of SPFinder

SPFinder is a tool for retrieving the missing links to patches of CVEs. Given the text description of the CVE, SPFinder casts this problem as retrieving the commit (commit message, code diff) which most closely match the CVE description. The backbone of SPFinder are GritLM [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) and CodeXEmbed [CodeXEmbed](https://huggingface.co/Salesforce/SFR-Embedding-Mistral). It leverages a hierarchical embedding technique to extend the context length for representing a commit. It further proposes a three-phase framework for improving the scalability. SPFinder outperforms existing work: [PatchScout](https://yuanxzhang.github.io/paper/patchscout-ccs21.pdf), [PatchFinder](https://dl.acm.org/doi/10.1145/3650212.3680305), and [VFCFinder](https://dl.acm.org/doi/pdf/10.1145/3634737.3657007) on two datasets (GitHubAD and PatchFinder), it further outperforms the MRR and Recall@10 of VoyageAI, a commercial embedding API with state-of-the-art performance, by 18\% and 28\% on our the two datasets. 

## Pipeline of SPFinder

First, given the a CVE description and all commits (commit message + code diff) of a repository, SPFinder first pre-ranks all commits using BM25 + CVE time information; then, for the top 10k commits, it extracts 9 features(Table~\ref{tab:feature_set}) including hierarchical embedding and path embedding; finally, it leverages LightGBM to combine these features into the final ranking score.


## Features used in SPFinder

Given the CVE description and each commit, SPFinder uses the following feature groups to compute the final similarity score:

| Feature Group      | Feature                                                              |
|--------------------|----------------------------------------------------------------------|
| **Code embedding** | 1. GritLM cosine with truncated diff                                 |
|                    | 2. Max GritLM cosine with all files in diff                          |
|                    | 3. GritLM cosine with mean of top-1 vectors of all files in diff     |
|                    | 4. GritLM cosine with mean of top-2 vectors of all files in diff     |
| **BM25**           | 5. BM25 ElasticSearch                                                |
| **Time**           | 6. \#commits between CVE reserve time and commit                     |
|                    | 7. \#commits between CVE publish time and commit                     |
| **Path**           | 8. Jaccard Index between NER-paths and commit-paths                  |
|                    | 9. Voyage AI~\cite{voyage} cosine between NER-paths and commit-paths |



## Instruction of reproducing SPFinder

To reproduce SPFinder, first, you need to collect the dataset following the `README.md` under data_prepare. This will creates the following directories: 

* `repo2commits_diff/`, which stores all the commit and diff data
* `csv/AD_train.csv` and `csv/AD_test.csv`, which stores the patch of each CVE

You can then reproduce ScalPatchFinder by following the steps below. Notice that we store the output of all steps under `feature/method_name/result`, including the baselines:

* `BM25+Time with ElasticSearch (Section 3.2)`: first, pre-rank all commits following the `README.md` under `es`, the output of this step is saved under `feature/bm25_time/result`

* `hierarchical embedding (Section 3.3)`: next, run code embedding following the `README.md` under `embedding`, the output of this step is saved under `feature/grit_instruct/result` (i.e., feature 1), `feature/grit_instruct_512_file/result_head1` (i.e., feature 3), `feature/grit_instruct_512_file/result_head2` (i.e., feature 4), and `feature/grit_instruct_512_file/result_head5_max` (i.e., feature 2). 

* `ner + in-repo search (Section 3.4)`: next, augment the file paths to bridge the CVE-patch gap by following the `README.md` under `ner_inrepo`, the output of this step is saved under `feature/path/result` (i.e., feature 9)

* `learning to rank (Section 3.5)`: finally, aggregate all features using lightgbm's LambdaRank algorithm by following the `README.md` under `ltr`, the output of this step is saved under `feature/ltr/result`. 

## Instruction of evaluating the ranking score of SPFinder and baselines

After the similarities are stored, you can evaluate the ranking score of each method using 

```
python recall.py --model_name {model_name} --dataset_name {dataset_name} --suffix {suffix}
```

An example of how to set up the arguments in recall.py can be found in run_recall.sh, where suffix can be set as empty (feature 1), `_head5_max` (feature 2), `_head1` (feature 3), `_head2` (feature 4).

which will reproduce the following Table 4 results with the following model_name:

* `voyage`: voyage commit level indexing
* `bm25_time`: BM25+time
* `patchscout`: PatchScout (Tan et al. 2021) 
* `patchfinder`: PatchFinder (Li et al. 2024) 
* `ltr`: ScalPatchFinder
* `grit_instruct_512_file`: grit_head2 when suffix is `_head2` 





<p align="center">
    <img src="figs/table5.png"  width=800>
    <br>
</p>
