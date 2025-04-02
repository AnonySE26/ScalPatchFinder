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
from functools import partial
import sys

sys.path.append("../../")
from recall import *



def create_ranked_commits_files(cve_groups, output_dir, feature_name='ltr_score', tmp=False):
    for cve, group in tqdm(cve_groups, desc=f"Creating ranked commits using {feature_name}"):
        sorted_group = group.sort_values(feature_name, ascending=False)
        ranked_commits = {
            row["commit_id"]: {"new_score": float(row[feature_name])}
            for _, row in sorted_group.iterrows()
        }
        if tmp:
            result_folder = output_dir
        else:
            row0 = group.iloc[0]
            repo = row0['repo']
            result_folder = os.path.join('../../feature', f"{repo}", 'ltr', "result")
        os.makedirs(result_folder, exist_ok=True)
        output_file = os.path.join(result_folder, f"{cve}.json")
        with open(output_file, 'w') as f:
            json.dump(ranked_commits, f, indent=2)


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



def flaml_objective(train_df, valid_df, config):
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
    
    valid_df_with_scores = apply_lightgbm_model(valid_df, model, features)
    tmp_output_dir = "tmp_ranked_commits"
    os.makedirs(tmp_output_dir, exist_ok=True)
    valid_df_groupby_cve = valid_df_with_scores.groupby('cve')
    
    create_ranked_commits_files(valid_df_groupby_cve, tmp_output_dir, 'ltr_score', tmp=True)
    recall_results = calculate_tmp_valid_recall(valid_df_groupby_cve, tmp_output_dir)
    print(f"Recall results: {recall_results}")
    score = recall_results.get("ndcg10", 0)
    
    shutil.rmtree(tmp_output_dir)
    
    tune.report(score=score)


def main():
    global train_df, test_df
    
    parser = argparse.ArgumentParser(description='LightGBM Learning to Rank for commit ranking')
    parser.add_argument('--train_data', required=True, help='Path to training CSV file')
    parser.add_argument('--test_data', required=True, help='Path to test CSV file')
    parser.add_argument('--output_dir', default='./ranked_commits_lightgbm', help='Output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading training data from {args.train_data}...")
    train_df = pd.read_csv(args.train_data)
    
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    

    
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

    # randomly split train into 80% train, 20% valid
    # train_df -> train_df_80, valid_df_20
    # -------------------------------
    # FLAML
    # -------------------------------
    
    train_df_80 = train_df.groupby('cve', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=42))
    valid_df_20 = train_df.drop(train_df_80.index)
    search_space = {
        "learning_rate": tune.loguniform(0.005, 0.1),
        "num_leaves": tune.qrandint(10, 30, q=1),
        "min_data_in_leaf": tune.qrandint(5, 30, q=1)
    }
    
    print("Starting FLAML hyperparameter tuning...")
    analysis = tune.run(
        partial(flaml_objective, train_df_80, valid_df_20), 
        config=search_space,
        metric="score",        # flaml_objective metric
        mode="max",            # recall is higher the better
        num_samples=-1,        # no limit
        time_budget_s=3,       # 1 hour
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
    
    test_df_with_scores = apply_lightgbm_model(test_df, model, features)
    
    lightgbm_output_dir = os.path.join(args.output_dir, 'ranked_commits')
    os.makedirs(lightgbm_output_dir, exist_ok=True)
    create_ranked_commits_files(test_df_with_scores.groupby('cve'), lightgbm_output_dir, 'ltr_score')

    # output ltr output to ../../feature/owner@@repo/ltr/result/
    

if __name__ == "__main__":
    main()

    # python ltr.py --train_data ../../feature/final_feature/patchfinder/patchfinder_train_feature_v3.csv --test_data ../../feature/final_feature/patchfinder/patchfinder_test_feature_v3.csv
    # python ltr.py --train_data ../../feature/final_feature/AD/AD_train_feature_v4.csv --test_data ../../feature/final_feature/AD/AD_test_feature_v4.csv
