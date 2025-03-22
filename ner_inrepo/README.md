============================================================
                       In-Repo Search
============================================================
linux_elasticsearch/filepath_trace

------------------------------------------------------------
1. commits_file_path_1.py
------------------------------------------------------------
执行命令：
  python3 commits_file_path_1.py patchfinder train

说明：
  把 ../repo2commits_diff/{owner}@@{repo}.json 按文件名分开并保存，
  输出 feature 里的 path 目录下的 commits_file_{owner}@@{repo}.csv，
  然后合并并过滤为 "./commits_file_path_{dataset}_{type}.csv"

------------------------------------------------------------
2. NER 关键词提取
------------------------------------------------------------
说明：
  ner 文件夹中，每个 CVE 通过 GPT 提取出关键词,结果保存在ner文件夹里

python3 ner_prompt.py

------------------------------------------------------------
3. path_in_repo_search_2.py / path_in_repo_search_4.py
------------------------------------------------------------
执行命令：
  python path_in_repo_search_4.py --dataset patchfinder --mode multi --processes 4

说明：
  用 GitHub API 把关键词做 in repo search，
  找出前 k 个 file path，输出 cve2name_with_paths_{dataset}.csv

------------------------------------------------------------
4. compute_similarity_3.py
------------------------------------------------------------
输入文件：
  cve_file         = "./cve2name_with_paths_{dataset}.csv"
  commit_file_train= "./commits_file_path_{dataset}_train.csv"
  commit_file_test = "./commits_file_path_{dataset}_test.csv"

输出文件：
  cve_embedding_file    = "./cve_path_embeddings_{dataset}.json"
  commit_embedding_file = "./commit_path_embeddings_{dataset}.json"
  
说明：
  两个 voyage embedding 算出 cosine similarity，
  最终输出 output_file = "./cve_commit_path_similarity_{dataset}.csv"
