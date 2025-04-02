## Section 3.4: Bridging CVE-Patch Gap using NER and In-Repo Search

Navigate to `anonyICSE26/ner_inrepo/filepath_trace` and perform Named Entity Recognition (NER) combined with in-repo search to bridge the gap between CVEs and corresponding patches.

Install dependencies first:

```bash
conda create --name ner_inrepo python==3.10
pip install -r requirements.txt
```

### Step 1: File Path Extraction

Execute the following command to separate JSON commit files by file paths and save outputs:

```bash
python3 commits_file_path_1.py <dataset> <split>
```

Outputs are saved as CSV files under the `result` directory (`./result/commits_file_path_{dataset}_{type}.csv`).

### Step 2: NER Keyword Extraction

Extract keywords from each CVE description using GPT-based NER. Run:

```bash
python3 ner_prompt.py <dataset> <split>
```

Results `<dataset>_<split>_path.csv` will be saved in the `result` folder. You can directly use the output files by ignoring this step.

### Step 3: In-Repo Search

Perform GitHub API searches within repositories using extracted NER entities:

```bash
python path_in_repo_search_2.py --dataset <dataset> --mode multi --processes 4 --split <split>
```

Outputs CSV file: `result/cve2name_with_paths_{dataset}.csv`. Note: GitHub API uses multiple tokens to avoid rate limits; adjust tokens accordingly.

### Step 4: Feature Embedding

Generate embeddings from extracted features (paths) using Voyage embeddings:

```bash
python generate_feature_embed_3.py <dataset> <split>
```

Adjust Voyage tokens in the code (`generate_feature_embed_3.py`) as needed based on rate limits and available tokens.

Embeddings are stored within repository-specific directories under `anonyICSE26/feature/{owner}@@{repo}/path/`.

This section prepares the dataset with enriched path and entity embeddings for subsequent model training.

### Step 5: Path Similarity (Jaccard + SequenceMatcher)

Compute similarity scores between CVE and commit file paths using a combination of Jaccard similarity and SequenceMatcher:

```bash
python jaccard_SequenceMatcher.py <dataset> <split>
```

Outputs similarity scores in JSON format under `anonyICSE26/feature/{owner}@@{repo}/jaccard/result/{cve}.json` directory, preparing detailed similarity-based features for further training.

### Step 6: Path Similarity (Embedding)

To compute the path similarity using voyage, you need to navigate to `anonyICSE26/embedding`. Use the embedding conda env and compute_sim.py to compute the similarity by setting the model_name to `path`:

```bash
python compute_sim.py \
--model_name path \
--dataset_name <dataset_name> \
--is_file \
--head_count 1 \
--is_train \
--mode mean 
```

You can also directly run compute_sim.sh

Output similarity scores in JSON format under `anonyICSE26/feature/{owner}@@{repo}/path/result/{cve}.json` directory, preparing detailed similarity-based features for further training.
