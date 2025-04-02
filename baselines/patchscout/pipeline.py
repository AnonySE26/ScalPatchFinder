import time
import torch
import pathlib
import itertools

import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data import load_code_data, get_commit_list, get_patchscout_features

LABEL_PATH = pathlib.Path("../../csv")
FEATURE_PATH = pathlib.Path("../../feature")
DATASETS_PATH = pathlib.Path("../../repo2commits_diff")
FEATURE_COLUMNS = [
    'cve_match', 'bug_match',  # VI
    'func_same_cnt', 'func_same_ratio', 'func_unrelated_cnt',  # VL
    'filepath_same_cnt', 'filepath_same_ratio', 'filepath_unrelated_cnt',  # VL
    'file_same_cnt', 'file_same_ratio', 'file_unrelated_cnt',  # VL
    'patch_like', 'vuln_type_1', 'vuln_type_2', 'vuln_type_3',  # VT
    'msg_shared_num', 'msg_shared_ratio', 'msg_max', 'msg_sum', 'msg_mean', 'msg_var',  # VDT
    'code_shared_num', 'code_shared_ratio', 'code_max', 'code_sum', 'code_mean', 'code_var'  # VDT
]
NUM_FEATURES = len(FEATURE_COLUMNS)

class RankNet(nn.Module):
    def __init__(self, num_feature):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_feature, 32),
            nn.Linear(32, 16), nn.Linear(16, 1))
        self.output_sig = nn.Sigmoid()
        # self.output_sig = nn.Sigmoid()
        # self.out = nn.Linear(16, 1)

    def forward(self, input_1, input_2):
        s1 = self.model(input_1)
        s2 = self.model(input_2)
        out = self.output_sig((s1 - s2))
        return out

    def predict(self, input_):
        s = self.model(input_)
        return s


class PairDataset(Dataset):
    def __init__(self, df, max_negatives=5000):
        self.pairs = []
        self.labels = []

        for cve, group in df.groupby("cve"):
            pos_samples = group[group["label"] == 1][FEATURE_COLUMNS].values
            neg_samples = group[group["label"] == 0][FEATURE_COLUMNS].values

            # limit the number of negatives
            if len(neg_samples) > max_negatives:
                neg_samples = neg_samples[:max_negatives]

            # generate pair dataset
            for pos, neg in itertools.product(pos_samples, neg_samples):
                self.pairs.append((pos, neg))
                self.labels.append(1)  # 1 means pos should rank higher than neg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos, neg = self.pairs[idx]
        return torch.tensor(pos, dtype=torch.float32),\
               torch.tensor(neg, dtype=torch.float32),\
               torch.tensor(self.labels[idx], dtype=torch.float32)


def prepare_data(df, n=1000, data_base_path=pathlib.Path("."), split="train"):
    """

    :param df: a dataframe with ['cve', 'owner', 'repo', 'patch', 'name', 'cvedesc']
    :param n: number of data to sample
    :param data_base_path:
    :param split:
    :return:
    """
    assert split in ["train", "test"]

    DATA_PATH = data_base_path / split
    DATA_PATH.mkdir(exist_ok=True, parents=True)

    for name, group in tqdm(df.groupby("name"), total=df["name"].nunique(), desc="Processing Repos"):
        print(f"Processing {name}...")
        filename = DATA_PATH / f"{name}.pkl"
        if filename.exists():
            continue

        try:
            owner_name, repo_name = name.split("@@")
            code_df = load_code_data(owner_name, repo_name, data_path=DATASETS_PATH).rename(columns={"msg_token": "commit_msg", "diff_token": "diff"})
        except Exception:
            continue

        ##################################################
        # APPROXIMATION: consider up to N commits for each CVE

        # initialize a new list for each repository
        sampled_code_dfs = list()

        for cve, ggroup in group.groupby("cve"):
            # make sure there are n commit_ids per CVE, it may or may not contain the positive IDs
            sorted_commit_ids = get_commit_list(FEATURE_PATH, owner_name, repo_name, [cve], BM25_K=n)

            # for either case: len(positive_commit_ids) + len(negative_commit_ids) == n
            if split == "train":
                positive_commit_ids = ggroup.patch.tolist()
                negative_commit_ids = [commit_id for commit_id in sorted_commit_ids if commit_id not in positive_commit_ids][:-len(positive_commit_ids)]
            else:
                positive_commit_ids = [commit_id for commit_id in sorted_commit_ids if commit_id in ggroup.patch.tolist()]
                negative_commit_ids = [commit_id for commit_id in sorted_commit_ids if commit_id not in positive_commit_ids]

            sampled_code_df = pd.concat(
                [
                    code_df[code_df.commit_id.isin(positive_commit_ids)].assign(label=1),
                    code_df[code_df.commit_id.isin(negative_commit_ids)].assign(label=0)
                ]
            ).sample(frac=1)

            metadata_dict = ggroup.drop(columns=["patch"]).drop_duplicates().to_dict("records")[0]
            sampled_code_df = sampled_code_df.assign(**metadata_dict)
            sampled_code_dfs.append(sampled_code_df)

        all_df = pd.concat(sampled_code_dfs).reset_index(drop=True)
        print("Data Preparation Completed...")

        ##################################################
        # ACTUALLY RUN THE DATA PROCESSING
        try:
            processed_all_df = get_patchscout_features(all_df).reset_index(drop=True)
            processed_all_df.to_pickle(filename)
            print("Feature Generation Completed...")
        except Exception:
            print("Failure")


def train_patchscout(train_df, num_features, model_save_path, batch_size=8192):
    lr = 0.0001
    num_workers = 10
    num_epochs = 20
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RankNet(num_features).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = PairDataset(train_df)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    print("Training PatchScout...")
    print(time.strftime("%H:%M:%S"))

    for epoch in range(num_epochs):
        model.train()
        t1 = time.time()
        for data1, data2, label in train_dataloader:
            data1, data2, label = data1.to(device), data2.to(device), label.to(device)

            pred = model(data1, data2)
            loss = criterion(pred, label.unsqueeze(1).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time.time()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Time {int(t2 - t1)}s, Loss: {loss.item():.10f}, Lr: {lr:.4f}')

    # Save the trained model
    torch.save(model.cpu().state_dict(), model_save_path)
    print(f'Model saved at {model_save_path}')
    return model


def test_patchscout(test_df, num_features, model_path, batch_size=8192):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RankNet(num_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ranking_results = []
    grouped_test = test_df.groupby("cve")

    with torch.no_grad():
        for cve, group in tqdm(grouped_test, desc="Processing CVEs", unit="CVE"):
            commit_ids = group["commit_id"].tolist()
            features = torch.tensor(group[FEATURE_COLUMNS].values, dtype=torch.float32, device=device)

            num_commits = len(commit_ids)
            if num_commits < 2:
                ranking_results.append((cve, commit_ids[0], 0, 1))
                continue

            scores = torch.zeros(num_commits, dtype=torch.float32, device=device)

            # generate all possible index pairs
            indices = list(itertools.combinations(range(num_commits), 2))
            indices = torch.tensor(indices, device=device)
            idx1, idx2 = indices[:, 0], indices[:, 1]

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = slice(batch_start, batch_start + batch_size)
                data1, data2 = features[idx1[batch_indices]], features[idx2[batch_indices]]

                pred_batch = model(data1, data2).squeeze()

                # compute win rates
                scores[idx1[batch_indices]] += (pred_batch > 0.5).float()
                scores[idx2[batch_indices]] += (pred_batch <= 0.5).float()

            # get results
            sorted_indices = torch.argsort(-scores)
            sorted_commits = [commit_ids[i] for i in sorted_indices.tolist()]
            sorted_scores = scores[sorted_indices].tolist()

            ranking_results.extend((cve, commit_id, score, rank) for rank, (commit_id, score) in enumerate(zip(sorted_commits, sorted_scores), start=1))

    return pd.DataFrame(ranking_results, columns=["cve", "commit_id", "score", "rank"])