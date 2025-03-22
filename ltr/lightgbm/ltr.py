import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import shutil
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from flaml import tune
import shap
import matplotlib.pyplot as plt
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


def prepare_training_data(train_df, valid_list_df):
    print("Preparing training data...")
    
    relevant_pairs = set()
    for _, row in valid_list_df.iterrows():
        relevant_pairs.add((row['cve'], row['patch']))
    
    # train_df['label'] = 0
    for idx, row in tqdm(train_df.iterrows(), desc="Labeling data", total=len(train_df)):
        if (row['cve'], row['commit_id']) in relevant_pairs:
            train_df.at[idx, 'label'] = 1
    
    print(f"Total training samples: {len(train_df)}")
    print(f"Relevant samples (label=1): {train_df['label'].sum()}")
    print(f"Non-relevant samples (label=0): {len(train_df) - train_df['label'].sum()}")
    
    return train_df

def train_lightgbm_model(train_df, model_file, params=None):
    print("Training LightGBM ranking model...")
    
    features = ['grit_head1', 'grit_head2', 'grit', 'grit_max', 'bm25', 'reserve_time_diff', 'publish_time_diff', 'path', 'jaccard']
    # features = ['bm25']
    available_features = [f for f in features if f in train_df.columns]
    
    print(f"Using features: {available_features}")
    
    for feature in available_features:
        if train_df[feature].isna().any():
            print(f"Filling missing values in {feature}")
            train_df[feature] = train_df[feature].fillna(0)
    
    X = train_df[available_features].values
    y = train_df['label'].values
    
    le = LabelEncoder()
    cve_encoded = le.fit_transform(train_df['cve'])
    
    groups = []
    current_group = []
    
    max_group_size = 8000
    
    for i, cve in enumerate(train_df['cve']):
        if i > 0 and cve != train_df['cve'].iloc[i-1]:
            if len(current_group) > max_group_size:
                num_splits = int(np.ceil(len(current_group) / max_group_size))
                for j in range(num_splits):
                    groups.append(min(max_group_size, len(current_group) - j * max_group_size))
            else:
                groups.append(len(current_group))
            current_group = []
        current_group.append(i)
    
    if current_group:
        if len(current_group) > max_group_size:
            num_splits = int(np.ceil(len(current_group) / max_group_size))
            for j in range(num_splits):
                groups.append(min(max_group_size, len(current_group) - j * max_group_size))
        else:
            groups.append(len(current_group))
    
    if params is None:
        params = {
            'objective': 'lambdarank',
            'metric': ['ndcg', 'map'],
            'eval_at': [10, 100, 1000],
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'verbose': 1
        }
    
    train_data = lgb.Dataset(X, label=y, group=groups)
    
    model = lgb.train(params, train_data, num_boost_round=100)
    
    model.save_model(model_file)
    print(f"LightGBM model saved to {model_file}")
    
    importance = model.feature_importance()
    importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(importance_df)
    
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_df[available_features])


    plt.figure()
    shap.summary_plot(shap_values, train_df[available_features], show=False)
    # plt.xticks(fontsize=17)
    # plt.yticks(fontsize=18)
    # plt.xlabel("SHAP Value", fontsize=18)
    
    plt.savefig("./shap/shap_summary_detailed.png", bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved")
    
    return model, le, available_features

def apply_lightgbm_model(test_df, model, features):
    print("Applying LightGBM model to test data...")
    
    test_df_with_scores = test_df.copy()
    
    for feature in features:
        if feature in test_df_with_scores.columns and test_df_with_scores[feature].isna().any():
            print(f"Filling missing values in {feature}")
            test_df_with_scores[feature] = test_df_with_scores[feature].fillna(0)
    
    for cve, group in tqdm(test_df_with_scores.groupby('cve'), desc="Scoring"):
        if all(feature in group.columns for feature in features):
            X_test = group[features].values
            
            scores = model.predict(X_test)
            
            for i, idx in enumerate(group.index):
                test_df_with_scores.at[idx, 'ltr_score'] = scores[i]
        else:
            for idx in group.index:
                test_df_with_scores.at[idx, 'ltr_score'] = test_df_with_scores.at[idx, 'grit_head2']
    
    return test_df_with_scores



def flaml_objective(config):
    params = {
        'objective': 'lambdarank',
        'metric': ['ndcg'], # ['ndcg', 'recall', 'mrr'],
        'eval_at': [10, 100, 1000], # [10, 100, 1000],
        'learning_rate': config["learning_rate"],
        'num_leaves': config["num_leaves"],
        'min_data_in_leaf': config["min_data_in_leaf"],
        'feature_fraction': 0.8,
        'verbose': 1
    }
    
    tmp_model_file = "tmp_lightgbm_model.txt"
    model, le, features = train_lightgbm_model(train_df, tmp_model_file, params)
    
    test_df_with_scores = apply_lightgbm_model(test_df, model, features)
    
    tmp_output_dir = "tmp_ranked_commits"
    os.makedirs(tmp_output_dir, exist_ok=True)
    create_ranked_commits_files(test_df_with_scores, tmp_output_dir, 'ltr_score')
    
    recall_results = calculate_overall_recall(args_valid, tmp_output_dir)
    score = recall_results.get("ndcg10", 0)
    
    shutil.rmtree(tmp_output_dir)
    
    tune.report(score=score)


def main():
    global train_df, test_df, args_valid
    
    parser = argparse.ArgumentParser(description='LightGBM Learning to Rank for commit ranking')
    parser.add_argument('--train_data', required=True, help='Path to training CSV file')
    parser.add_argument('--test_data', required=True, help='Path to test CSV file')
    parser.add_argument('--valid_list', required=True, help='Path to validation list CSV')
    parser.add_argument('--output_dir', default='./ranked_commits_lightgbm', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for LightGBM')
    parser.add_argument('--num_leaves', type=int, default=31, help='Num leaves for LightGBM')
    parser.add_argument('--num_boost_round', type=int, default=100, help='Number of boosting rounds')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading training data from {args.train_data}...")
    train_df = pd.read_csv(args.train_data)
    
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    print(f"Loading validation list from {args.valid_list}...")
    valid_list_df = pd.read_csv(args.valid_list)
    
    train_df = prepare_training_data(train_df, valid_list_df)
    
    if 'publish_time_diff' in train_df.columns:
        train_df['publish_time_diff'] = train_df['publish_time_diff'] * 0.5
    if 'reserve_time_diff' in train_df.columns:
        train_df['reserve_time_diff'] = train_df['reserve_time_diff'] * 0.5

    if 'publish_time_diff' in test_df.columns:
        test_df['publish_time_diff'] = test_df['publish_time_diff'] * 0.5
    if 'reserve_time_diff' in test_df.columns:
        test_df['reserve_time_diff'] = test_df['reserve_time_diff'] * 0.5
    
    if "path" in train_df.columns:
        train_df['path'] = train_df['path']
    if "path" in test_df.columns:
        test_df['path'] = test_df['path']

    if "jaccard" in train_df.columns:
        train_df['jaccard'] = train_df['jaccard']
    if "jaccard" in test_df.columns:
        test_df['jaccard'] = test_df['jaccard']

    args_valid = args.valid_list

    # -------------------------------
    # FLAML
    # -------------------------------
    search_space = {
        "learning_rate": tune.loguniform(0.005, 0.1),
        "num_leaves": tune.qrandint(10, 30, q=1),
        "min_data_in_leaf": tune.qrandint(5, 30, q=1)
    }
    
    print("Starting FLAML hyperparameter tuning...")
    analysis = tune.run(
        flaml_objective, 
        config=search_space,
        metric="score",        # flaml_objective metric
        mode="max",            # recall is higher the better
        num_samples=-1,        # no limit
        time_budget_s=3600,       # 1 hour
        resources_per_trial={"cpu": 8},
        low_cost_partial_config={
            "learning_rate": 0.01,
            "num_leaves": 15,
            "min_data_in_leaf": 10
        }
    )
    best_config = analysis.best_config
    
    print("Best hyperparameters found:", best_config, flush=True)
    
    # import pdb; pdb.set_trace()
    
    # -------------------------------
    # best_config 
    # -------------------------------
    
    params = {
        'objective': 'lambdarank',
        'metric': ['ndcg', 'recall', 'mrr'],
        'eval_at': [10, 100, 1000],
        'learning_rate': best_config['learning_rate'],
        'num_leaves': best_config['num_leaves'], 
        'min_data_in_leaf': best_config["min_data_in_leaf"],
        'feature_fraction': 0.8,
        'verbose': 1
    }
    
    # params = {
    #     'objective': 'lambdarank',
    #     'metric': ['ndcg', 'recall', 'mrr'],
    #     'eval_at': [10, 100, 1000],
    #     'learning_rate': 0.01,
    #     'num_leaves': 30, 
    #     'min_data_in_leaf': 38,
    #     'feature_fraction': 0.8,
    #     'verbose': 1
    # }
        
    model_file = os.path.join(args.output_dir, "lightgbm_model.txt")
    model, le, features = train_lightgbm_model(train_df, model_file, params)
    
    with open(os.path.join(args.output_dir, "features.json"), "w") as f:
        json.dump({"features": features}, f, indent=2)
    
    test_df_with_scores = apply_lightgbm_model(test_df, model, features)
    
    lightgbm_output_dir = os.path.join(args.output_dir, 'ranked_commits')
    os.makedirs(lightgbm_output_dir, exist_ok=True)
    create_ranked_commits_files(test_df_with_scores, lightgbm_output_dir, 'ltr_score')
    
    recall_results = calculate_overall_recall(args.valid_list, lightgbm_output_dir)
    
    repo_recall_results = calculate_recall_by_repo(args.valid_list, lightgbm_output_dir)
    
    output_path = os.path.join(args.output_dir, "lightgbm_recall_results.json")
    with open(output_path, "w") as f:
        json.dump({"recall_at_k": recall_results}, f, indent=4)
    
    print(f"Overall results saved to {output_path}")
    
    repo_output_path = os.path.join(args.output_dir, "lightgbm_recall_by_repo_results.json")
    with open(repo_output_path, "w") as f:
        json.dump(repo_recall_results, f, indent=4)
    
    print(f"Repository-specific results saved to {repo_output_path}")
    
    repo_csv_path = os.path.join(args.output_dir, "lightgbm_recall_by_repo_results.csv")
    
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
    # python ltr.py --train_data ../../feature/final_feature/patchfinder/patchfinder_train_feature_v2.csv --test_data ../../feature/final_feature/patchfinder/patchfinder_test_feature_v2.csv --valid_list ../../csv/patchfinder_test.csv
    # python ltr.py --train_data ../../feature/final_feature/AD/AD_train_feature_v3.csv --test_data ../../feature/final_feature/AD/AD_test_feature_v3.csv --valid_list ../../csv/AD_test.csv