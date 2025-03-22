## Section 3.5: Learning to Rank

Navigate to `anonyICSE26/ltr` and train a ranking model using LightGBM to further refine commit rankings based on multiple features:

### Training and Evaluation

Run the following command to train and evaluate the LightGBM ranking model with hyperparameters optimized by FLAML:
```bash
python ltr.py --train_data <path_to_training_csv> --test_data <path_to_test_csv> --valid_list <path_to_validation_csv>
```

### Important Notes:
- If running on Mac, first install OpenMP:
```bash
brew install libomp
```

- Hyperparameters such as learning rate, number of leaves, and number of boosting rounds can be adjusted directly in `ltr.py`.

- Outputs including ranked commits, overall recall metrics, and repository-specific recall metrics will be stored in the specified output directory.
