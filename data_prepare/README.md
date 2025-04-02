# Preparing the data

```bash
conda create --name data_prepare python==3.10
pip install -r requirements.txt
```

The CVE-to-patch data can be found under `../csv/`

## Collecting the commit message and diff data 

Run the following command to prepare commit messages and diffs:

```bash
python prepare_data.py
```

Datasets include `AD` and `patchfinder`, with splits into `train` and `test` sets.

## Selecting the hard repos as the test data (Section 4.2)

We first select the harder repos in the data patchfinder (`../csv/patchfinder.csv`) as its test data, then we remove the repeated repos from AD, and select the harder repos as AD's test data. The remaining repos in each dataset become the training data.  

```
python select_test_github.py patchfinder # first split the train/test data of patchfinder
python select_test_github.py AD # then split the train/test data, excluding the repos already in patchfinder_test.csv
```

## Selecting the training data 

Due to the cost for indexing more commits, we sample a small set of the training data as explained in Section 3.5:

```
python select_train.py AD
```
