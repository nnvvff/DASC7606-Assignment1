# HKU-DASC7606-A1

HKU DASC 7606 Assignment 1 (Computer Vision: Image Classification), 2026–2027.

This codebase is only for the HKU DASC 7606 course in the 2026–2027 academic year. Please do not upload your answers or this codebase to any public platform (for example, GitHub) before permission is given. All rights reserved.

# 1 Introduction

## 1.1 Background: Image Classification
Image classification is a fundamental problem in computer vision, which involves assigning a label or category to an image. The goal is to develop a model that can automatically identify and classify images into different categories. The categories can be objects, actions, or scenes. For example, a model can be trained to classify images of animals, vehicles, or buildings into different categories.Since solutions to image classification problems based on deep learning have become highly mature, the main objective of this assignment is to help you become familiar with the complete workflow of configuring neural networks, including GPU usage, model training and testing, and network design.

## 1.2  What Will You Learn from This Assignment?
This assignment will guide you through the setup and use of a GPU cluster. Following the provided examples, you will learn how to train and test simple neural networks, design network architectures, and more. You will be required to implement and train a convolutional neural network (CNN) on your own. Additionally, you will explore methods to improve network performance, such as by incorporating batch normalization.

The goals of this assignment are as follows:

- Gain experience of implementing neural networks with a popular deep learning framework
PyTorch.
- Develop a deep learning system from scratch, including network design, model training,
hyperparameter tuning, training visualization, model inference and performance evaluation.


# 2 Setup

You can work on the assignment in one of two ways: locally on your own machine, or on a virtual machine on HKU GPU Farm.

## 2.1 Working remotely on HKU GPU Farm (Recommended)
Note: after following these instructions, make sure you go to work on the assignment below (i.e., you can skip the Working locally section).

As part of this course, you can use HKU GPU Farm for your assignments. We recommend you follow the quickstart provided by the [official website](https://www.cs.hku.hk/gpu-farm/quickstart) to get familiar with HKU GPU Farm.

After checking the quickstart document, make sure you have gained the following skills:
- Knowing how to access the GPU Farm and use GPUs in interactive mode. We recommend using GPU support for this assignment, since your training will go much, much faster.
- Getting familiar with running Jupyter Lab without starting a web browser.
- Knowing how to use tmux for unstable network connections.

## 2.2 Working locally on your own machine
If you have the GPU resources on your own PC/laptop and wish to use that, that’s fine – you’ll need to install the drivers for your GPU, install CUDA, install cuDNN, and then install PyTorch. You could theoretically do the entire assignment with no GPUs, though this will make training the model much slower.

## 2.3 Environment Setup

Install Python 3.10 and create a dedicated Conda environment:

```bash
git clone https://github.com/nnvvff/DASC7606-Assignment1.git
cd DASC7606-Assignment1
conda create -n cv_env python=3.10
conda activate cv_env
```

Install PyTorch 2.0.1 with CUDA 11.8:

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining packages used by the assignment:

```bash
pip install numpy matplotlib tqdm jupyterlab
```

From the repository directory, start JupyterLab and open `Assignment_1.ipynb`:

```bash
jupyter lab
```

You must use the provided environment and packages listed above. **Do not install, import, or depend on any additional third-party packages.** Your submitted `model.py` must load successfully in this environment. It may use the Python standard library, PyTorch, and torchvision, but it must not download packages, model weights, or other files when imported.

# 3 Working on the Assignment

## 3.1 Code and Data

Everything you need to do is provided in the [Jupyter notebook](Assignment_1.ipynb). Please follow the [HKU GPU Farm quickstart](https://www.cs.hku.hk/gpu-farm/quickstart#:~:text=Running%20Jupyter%20Lab%20without%20Starting%20a%20Web%20Browser) to set up JupyterLab.

The notebook downloads CIFAR-10 automatically when required. The unseen evaluation data is not distributed and will be evaluated separately by the teaching team. Both datasets use the input preprocessing and model interface specified in the notebook.

## 3.2 Assignment Tasks

**Task 1: Complete three experiments**

Complete all required code for the following three experiments in [Assignment_1.ipynb](Assignment_1.ipynb):

1. Build and train the specified convolutional neural network.
2. Add batch normalization and train the CNN with BN.
3. Design, train, and evaluate your own improved model.

For each experiment, report the required training and validation curves and evaluate the trained model as instructed in the notebook.

**Task 2: Write a report (no more than 2 pages)**

Your report should include three main sections: introduction, methods, and experiments and analysis.

## 3.3 Files to Submit

1. **Final report** (`report.pdf`, up to 2 pages)

   1. Introduction: briefly introduce the task, background, and related work.
   2. Methods: explain your improvements to the baseline model.
   3. Experiments and analysis: include at least three experiments and analyze the results. Possible analysis includes validation-set ablations, hyperparameters, model architectures, loss functions, and training/validation curves.

2. **Code**

   - `Assignment_1.ipynb`, including your completed code and outputs.
   - `model.py`, containing all code required to define your final model. It must expose a no-argument `build_model()` function that returns the exact architecture used to train the submitted checkpoint. Importing `model.py` must not start training, load a dataset, download files, or load the checkpoint automatically.
   - Any additional code files required by your notebook, if applicable.

3. **Model weights**

   - Submit the final checkpoint as `model.pt` when its file size is **100 MB or less**.
   - If `model.pt` is **larger than 100 MB**, submit `model_link.txt` instead. The file must contain an accessible Google Drive or Dropbox download link to the checkpoint.
   - Ensure that sharing permissions allow the teaching team to download the checkpoint without requesting access.
   - The submitted checkpoint must load strictly into the model returned by `build_model()` in `model.py`.

If your student ID is `30300xxxxx`, organize the Moodle submission as follows:

```text
30300xxxxx.zip
├── report.pdf
├── Assignment_1.ipynb
├── model.py
├── model.pt / model_link.txt
├── additional code files, if any
└── README.md             # Optional; include it if extra instructions are needed
```

Submit `model.pt` when the checkpoint is 100 MB or less. Submit `model_link.txt` instead only when `model.pt` is larger than 100 MB. Do not include both files. The notebook, `model.py`, and checkpoint must correspond to the same final model.

## 3.4 Timeline

The following timeline is the same for **Sections A and B**:

- **September 18, 2026 (Friday):** Assignment release.
- **October 18, 2026 (Sunday), 23:59 HKT (GMT+8):** Submission deadline.

Late submission policy:

- A 10% penalty applies to assignments submitted within 1 day after the deadline.
- A 20% penalty applies to assignments submitted within 2 days after the deadline.
- A 50% penalty applies to assignments submitted within 7 days after the deadline.
- A 100% penalty applies to assignments submitted more than 7 days after the deadline.

## 3.5 Need More Support?

For questions that may be relevant to other students, first check or post in one of the following places:

- [GitHub Issues for this assignment](https://github.com/nnvvff/DASC7606-Assignment1/issues)
- [Shared discussion document](https://docs.google.com/document/d/1q01iqQvupl_uVBY_UcugenknwfbtYwSSBee3wXY30gY/edit?usp=sharing)

# 4 Marking Scheme

The assignment has three components:

| Component | Weight |
| --- | ---: |
| Public CIFAR-10 test-set accuracy | 50% |
| Unseen-data accuracy | 30% |
| Final report | 20% |

The accuracy bands determine the percentage of marks awarded within each performance component:

| Percentage of component marks | Test-set accuracy | Unseen-data accuracy |
| ---: | ---: | ---: |
| 100% | ≥80% | ≥70% |
| 90% | ≥70% and <80% | ≥60% and <70% |
| 80% | ≥65% and <70% | ≥55% and <60% |
| 70% | ≥60% and <65% | ≥50% and <55% |
| 60% | ≥50% and <60% | ≥40% and <50% |
| 0% | <50% | <40% |

Complete all required code in the notebook to be eligible for full marks on the test-set component. Partial credit for incomplete code will be considered only rarely.

The final report is assessed mainly on the richness of its experiments and analysis:

| Report quality | Percentage of report marks |
| --- | ---: |
| Reasonable number of experiments with analysis | 90–100% |
| Basic analysis | 80–90% |
| Insufficient analysis | Below 80% |

# References

1. Krizhevsky, A., Sutskever, I., and Hinton, G. E. “ImageNet Classification with Deep Convolutional Neural Networks.” NeurIPS 2012. [Paper](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
2. He, K., Zhang, X., Ren, S., and Sun, J. “Deep Residual Learning for Image Recognition.” CVPR 2016. [Paper](https://arxiv.org/pdf/1512.03385)
3. Ioffe, S. and Szegedy, C. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” ICML 2015. [Paper](https://arxiv.org/pdf/1502.03167)
4. [Previous HKU DASC7606 Assignment 1 repository](https://github.com/TianshuoY/HKU-DASC7606-A1)
5. [Convolutional Model: Step by Step — DeepLearning.AI on Coursera](https://www.coursera.org/learn/convolutional-neural-networks/programming/4xt9A/convolutional-model-step-by-step)
