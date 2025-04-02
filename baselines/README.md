# ScalPatchFinder Baselines
## Overview
- Step 1: Creating Environments (see details in "Step-by-Step Guides").
- Step 2: Running the Baselines. Each baseline can be run using `cd <badeline>/;
  python run_<baseline>.py`. After execution, the following files will be 
  generated:
  - Aggregate File: `<baseline>/results/<baseline>.pkl`.
  - Per-CVE File: `<ROOT>/feature/<repo>/result/<cve>.json`. Each per-CVE file 
    is a JSON object with the format:

    ```json
    {
      "commit_id_1": {
        "new_score": 0.85
      },
      "commit_id_2": {
        "new_score": 0.67
      }
    }
    ```

    This format is compatible with the evaluation script `recall.py`.
- Step 3: Evaluation. To evaluate using the generated results, run `python 
<ROOT>/recall.py`. This script reads the per-CVE result files and computes 
  recall 
  metrics accordingly.

## Step-by-Step Guides

To ensure reproducibility, we recommend using a dedicated virtual 
environment for each baseline. Note that the `torch` version depends on your 
operating system and CUDA version. Follow the official [PyTorch 
installation guide](https://pytorch.org/get-started/previous-versions/) to 
install the correct version manually.

### PatchFinder
```bash
# step 1
cd patchfinder/
conda create --name patchfinder python==3.10
conda activate patchfinder
pip install -r requirements.txt
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# step 2
python run_patchfinder.py
```
### PatchScout
```bash
# step 1
cd patchscout/
conda create --name patchscout python==3.10
conda activate patchscout
pip install -r requirements.txt
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# step 2
python run_patchscout.py
```

### VFCFidner
```bash
# step 1
cd vfcfinder/
conda create --name vfcfinder python==3.10
conda activate vfcfinder
pip install -e .
pip install huggingface_hub==0.25 swifter

# step 2
python run_vfcfinder.py
```