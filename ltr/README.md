
## Section 3.5: Learning to Rank

Navigate to `anonyICSE26/ltr/lightgbm/` and train a ranking model using LightGBM to further refine commit rankings based on multiple features:

Install dependencies first:

```bash
conda create --name ltr python==3.10
pip install -r requirements.txt
```

### Combine the features into LTR data

After all features are computed (i.e., features under `anonyICSE25/feature/{owner}@@{repo}/{method_name}/result/`, to get the ltr train and test data based on all the features computed, run:
```
python extract_ltr_feature.py AD {split}
```
the output will be saved under anonyICSE25/feature/final_feature/{dataset_name}

### Training and Evaluation

Run the following command to train and evaluate the LightGBM ranking model with hyperparameters optimized by FLAML:
```bash
python ltr.py --train_data <path_to_training_csv> --test_data <path_to_test_csv>
```
For example, AD dataset:
```bash
python ltr.py --train_data ../../feature/final_feature/AD/AD_train_feature_v4.csv --test_data ../../feature/final_feature/AD/AD_test_feature_v4.csv
```

For patchfinder dataset:
```bash
python ltr.py --train_data ../../feature/final_feature/patchfinder/patchfinder_train_feature_v3.csv --test_data ../../feature/final_feature/patchfinder/patchfinder_test_feature_v3.csv
```

### Important Notes:
- If running on Mac, first install OpenMP:
```bash
brew install libomp
```

- Hyperparameters such as learning rate, number of leaves, and number of boosting rounds can be adjusted directly in `ltr.py`.

- Outputs including ranked commits, overall recall metrics, and repository-specific recall metrics will be stored in the specified output directory.
