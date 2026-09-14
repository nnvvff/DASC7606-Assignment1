# Assignment 1

## Environment Setup

Install **Python 3.8 or newer**. A Conda environment is recommended; if you do not use one, make sure all dependencies are installed in your Python environment.

To obtain the original assignment repository and create a Python 3.10 environment, run:

```bash
git clone https://github.com/VIOLINARTHUR/HKU-DASC7606-A1.git
conda create -n cv_env python=3.10
conda activate cv_env
```

Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your system. For PyTorch 2.0.1 with CUDA 11.8, use:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining packages needed to work with `Assignment_1.ipynb`:

```bash
pip install numpy matplotlib tqdm jupyterlab
```

From the directory containing the notebook, start JupyterLab with `jupyter lab` and open `Assignment_1.ipynb`.

## Marking Scheme

The assignment has three components:

| Component | Weight |
| --- | ---: |
| Test-set accuracy | 50% |
| Unseen-data accuracy | 30% |
| Final report | 20% |

The accuracy bands determine the percentage of marks awarded **within each performance component**:

| Percentage of component marks | Test-set accuracy | Unseen-data accuracy |
| ---: | ---: | ---: |
| 100% | ≥80% | ≥70% |
| 90% | ≥70% and <80% | ≥60% and <70% |
| 80% | ≥65% and <70% | ≥55% and <60% |
| 70% | ≥60% and <65% | ≥50% and <55% |
| 60% | ≥50% and <60% | ≥40% and <50% |
| 0% | <50% | <40% |

Complete all required code in the notebook to be eligible for full marks on the test-set component. Partial credit for incomplete code will be considered only rarely.

The **final report** is worth 20% of the assignment grade. It is assessed mainly on the richness of its experiments and analysis:

| Report quality | Percentage of report marks |
| --- | ---: |
| Reasonable number of experiments with analysis | 90–100% |
| Basic analysis | 80–90% |
| Insufficient analysis | Below 80% |
