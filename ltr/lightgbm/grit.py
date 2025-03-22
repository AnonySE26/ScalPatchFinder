import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import ndcg_score

def qrel2recall(qrel, total_relevant=None):
    
    mrr_val = 1.0 / (1 + [x for x in range(len(qrel)) if qrel[x] == 1][0])

    recallat = [10, 100, 500, 1000, 2000, 5000, 10000]

    recall = {k: sum(qrel[:k]) / total_relevant if total_relevant else 0 for k in recallat}
    ndcg = {k: ndcg_score([sorted(qrel, reverse=True)], [qrel], k=k) for k in recall.keys()}

    ret = {
        "mrr": mrr_val}
    for k in recall.keys():
        ret[f"recall{k}"] = recall[k]
        ret[f"ndcg{k}"] = ndcg[k]
    return ret

def get_recall_from_sortedcommits(ranked_commit_ids, patch):
    qrel = []
    ranked_commit_ids = ranked_commit_ids[:10000]
    for commit_id in ranked_commit_ids:
        if commit_id not in patch:
            qrel.append(0)
        else:
            qrel.append(1)
    if sum(qrel) == 0:
        return {
            "mrr": 0,
            "recall10": 0,
            "recall100": 0,
            "recall500": 0,
            "recall1000": 0,
            "recall2000": 0,
            "recall5000": 0,
            "recall10000": 0,
            "ndcg10": 0,
            "ndcg100": 0,
            "ndcg500": 0,
            "ndcg1000": 0,
            "ndcg2000": 0,
            "ndcg5000": 0,
            "ndcg10000": 0
        }
    return qrel2recall(qrel, len(patch))

def create_ranked_commits_files(df, output_dir, feature_name='grit_head2'):
    os.makedirs(output_dir, exist_ok=True)
    
    cve_groups = df.groupby('cve')
    
    for cve, group in tqdm(cve_groups, desc=f"Creating ranked commits using {feature_name}"):
        sorted_group = group.sort_values(feature_name, ascending=False)
        ranked_commits = {
            row["commit_id"]: {"new_score": float(row[feature_name])}
            for _, row in sorted_group.iterrows()
        }
        output_file = os.path.join(output_dir, f"{cve}.json")
        with open(output_file, 'w') as f:
            json.dump(ranked_commits, f, indent=2)

def calculate_overall_recall(valid_list_path, ranked_commits_dir):
    valid_list = pd.read_csv(valid_list_path)
    cve_patches = {}
    for cve, group in valid_list.groupby('cve'):
        cve_patches[cve] = group['patch'].tolist()
    recall_results = {}
    total_cves = 0
    
    for cve, patch in tqdm(cve_patches.items(), desc="Calculating Recall@k"):
        if not patch:
            continue
            
        cve_file_path = os.path.join(ranked_commits_dir, f"{cve}.json")
        if not os.path.exists(cve_file_path):
            continue
            
        with open(cve_file_path, "r") as f:
            ranked_commits = json.load(f)
        
        ranked_commit_ids = list(ranked_commits.keys())
        
        result = get_recall_from_sortedcommits(ranked_commit_ids, patch)
        if result is not None:
            total_cves += 1
            for k in result.keys():
                recall_results.setdefault(k, 0)
                recall_results[k] += result[k]
    total_cves = 952 # 629
    avg_recall_results = {k: recall_results[k] / total_cves for k in recall_results.keys()}
    
    print(f"\nOverall metrics (across {total_cves} CVEs):", flush=True)
    for k, value in avg_recall_results.items():
        print(f"{k}: {value:.4f}", flush=True)
    
    return avg_recall_results

def calculate_recall_by_repo(valid_list_path, ranked_commits_dir):
    valid_list = pd.read_csv(valid_list_path)
    
    repo_cves = {}
    for _, row in valid_list.iterrows():
        cve = row['cve']
        repo = row['repo']
        patch = row['patch']
        
        if repo not in repo_cves:
            repo_cves[repo] = {}
        
        if cve not in repo_cves[repo]:
            repo_cves[repo][cve] = []
        
        repo_cves[repo][cve].append(patch)
    
    repo_recall_results = {}
    
    for repo, cves in tqdm(repo_cves.items(), desc="Calculating Recall@k by repo"):
        repo_recall = {}
        total_cves = 0
        
        for cve, patches in cves.items():
            if not patches:
                continue
                
            cve_file_path = os.path.join(ranked_commits_dir, f"{cve}.json")
            if not os.path.exists(cve_file_path):
                continue
                
            with open(cve_file_path, "r") as f:
                ranked_commits = json.load(f)
            
            ranked_commit_ids = list(ranked_commits.keys())
            
            result = get_recall_from_sortedcommits(ranked_commit_ids, patches)
            if result is not None:
                total_cves += 1
                for k in result.keys():
                    repo_recall.setdefault(k, 0)
                    repo_recall[k] += result[k]
        
        if total_cves > 0:
            repo_recall = {k: repo_recall[k] / total_cves for k in repo_recall.keys()}
            repo_recall_results[repo] = repo_recall
    
    return repo_recall_results

def main():
    parser = argparse.ArgumentParser(description='Reproduce recall metrics using grit only')
    parser.add_argument('--test_data', required=True, help='Path to test CSV file')
    parser.add_argument('--valid_list', required=True, help='Path to validation list CSV')
    parser.add_argument('--output_dir', default='./ranked_commits_grit', help='Output directory for ranked commits')
    
    args = parser.parse_args()
    
    print(f"Loading test data from {args.test_data}...")
    df = pd.read_csv(args.test_data)
    
    create_ranked_commits_files(df, args.output_dir, 'grit_head2')
    
    recall_results = calculate_overall_recall(args.valid_list, args.output_dir)
    
    repo_recall_results = calculate_recall_by_repo(args.valid_list, args.output_dir)
    
    output_path = "grit_only_recall_results.json"
    with open(output_path, "w") as f:
        json.dump({"recall_at_k": recall_results}, f, indent=4)
    
    print(f"Overall results saved to {output_path}")
    
    repo_output_path = "grit_only_recall_by_repo_results.json"
    with open(repo_output_path, "w") as f:
        json.dump(repo_recall_results, f, indent=4)
    
    print(f"Repository-specific results saved to {repo_output_path}")
    
    repo_csv_path = "grit_only_recall_by_repo_results.csv"
    
    rows = []
    for repo, metrics in repo_recall_results.items():
        row = {"repo": repo}
        row.update(metrics)
        rows.append(row)
    
    repo_df = pd.DataFrame(rows)
    
    repo_order = [
        "tomcat", "mindsdb", "uaa", "opennms", "vantage6", "kubernetes", 
        "answer", "spring-framework", "directus"
    ]
    
    def custom_sort(repo):
        try:
            return repo_order.index(repo)
        except ValueError:
            return len(repo_order)
    
    repo_df = repo_df.sort_values(by="repo", key=lambda x: x.map(custom_sort))
    
    repo_df.to_csv(repo_csv_path, index=False)
    
    print(f"Repository-specific results saved to {repo_csv_path}")

if __name__ == "__main__":
    main()
    # python grit.py --test_data ../../feature/final_feature/patchfinder/patchfinder_test_feature.csv --valid_list ../../csv/patchfinder_test.csv --output_dir ./ranked_commits_lightgbm/ranked_commits_patchfinder
    # python grit.py --test_data ../../feature/final_feature/AD/AD_test_feature.csv --valid_list ../../csv/AD_test.csv --output_dir ./ranked_commits_lightgbm/ranked_commits