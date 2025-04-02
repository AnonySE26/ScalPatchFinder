# Hierarchical embedding with gritLM, indexing with gritLM/voyage

## Install

```
conda env remove -n embedding
conda env create -f environment.yml
conda activate embedding
pip install torch torchvision torchaudio triton gritlm
```

## Indexing commits with voyage/gritLM

Indexing the commits of the training data of the GitHub AD dataset using gritLM. Each commit is represented by the concatenation of commit message and diff. GritLM is truncated to 512 tokens; voyage is truncated to 50000 characters. 

```
python index_commits.py \
--model_name grit_instruct \
--dataset_name AD \
--is_train
```
You can also directly run the command line in submit_gpu_job.sh (slurm)

## Indexing files with gritLM

Indexing the each file of the commits GitHub AD dataset using gritLM. Each file is represented by the concatenation of the commit message and the file in the diff (e.g., `diff --git a/...`). GritLM is truncated to 512 tokens. 

```
python index_file.py \
--model_name grit_instruct_512_file \
--dataset_name AD \
--is_train train
```
You can also directly run the command line in submit_gpu_job.sh (slurm)

## Indexing queries with voyage/gritLM

```
python index_query.py \
--model_name grit_instruct_512_file \
--dataset_name AD \
--is_train train
```
You can also directly run the command line in submit_gpu_job.sh (slurm)


## Computing similarities based on the vector embeddings

Compute the cosine similarity with the indexed vector embeddings. `is_file` is whether the embeddings are file level or commit level. `head_count`, `mode` are arguments for aggregating the file-level hierarchical embedding (Section 3.2) and they are only useful when `is_file` is true: `head_count`: how many top-k BM25 vectors to keep for the aggregation, `mode`: whether to use their mean or max for computing the aggregated similarity score.

```
python compute_sim.py \
--model_name grit_instruct_512_file \
--dataset_name AD \
--is_file \
--head_count 5 \
--is_train \
--mode max 
```
You can also directly run compute_sim.sh

