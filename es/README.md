## Dataset Preparation (Prerequisite)

Run the following command to prepare commit messages and diffs from repositories located in `anonyICSE26/data_prepare`:
```bash
python prepare_data.py
```
Datasets include `AD` and `patchfinder`, with splits into `train` and `test` sets.

- **Elasticsearch Setup:**
  - Install and run Elasticsearch:
    ```bash
    ./elasticsearch-7.9.0/bin/elasticsearch
    ```
    Verify installation by accessing: `http://localhost:9200/`

  - Install Kibana for debugging via Dev Tools at `http://localhost:5601/`:
    ```bash
    ./kibana-7.9.0-linux-x86_64/bin/kibana
    ```

- **Increasing Elasticsearch Result Window:**
  Before running BM25 ranking, use Kibana Dev Tools:
  ```
  PUT /apache@@tomcat/_settings
  {
    "index": {
      "max_result_window": 80000
    }
  }
  ```

## Section 3.2: Pre-Ranking Large Repo using ElasticSearch and Time Distance

### 1. Elasticsearch Indexing

Navigate to `anonyICSE26/es` and index commit information and diffs for each repository into Elasticsearch. Perform indexing in two ways:
- **With file names:** Separately index each file.
- **Without file names:** Index without file separation.

### 2. Convert Git Logs to JSON

Convert commit messages and diffs from Git logs into JSON format for Elasticsearch indexing:
```bash
python convert_to_json.py <dataset>
```
The converted JSON data will be saved in:
```
./repo2commits_diff
```

### 3. Split JSON Data

To avoid Elasticsearch's 100MB file-size limitation, split the JSON data:
```bash
python split_multiple_repos.py <dataset>
```
Split files will be saved in:
```
repo2commits_diff/split_{repo}
```

### 4. Index JSON Data into Elasticsearch

- **General indexing:**
  ```bash
  python load_multiple_repos.py
  ```
  This will index all JSON data from `repo2commits_diff` into Elasticsearch.

- **File-name-specific indexing:** (no need to step 3 split files)
  ```bash
  python load_multiple_repos_files.py --mode <split> <dataset>
  ```

### 5. BM25 Ranking

Retrieve ranked commit lists for each CVE using BM25:
```bash
python bm25_rank_add_time_multirepos.py <dataset>
```
Results are stored in:
```
{dataset}_ranked_commits_bm25_time
```

For file-name-specific ranking:
```bash
python bm25_file_multirepos.py <dataset> <split>
```
Results are stored in:
```
./{dataset}_{mode}_bm25_diff_files
```

### 6. Weighted Re-ranking

Perform weighted re-ranking with TF-IDF, considering reserve and publish times:
```bash
python rerank_with_tfidf_add_time.py <bm25_weight> <reserve_time_weight> <publish_time_weight> <dataset>
```
Modify `input_dir` and `output_dir` in the script accordingly.

