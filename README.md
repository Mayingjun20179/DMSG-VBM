# DMSG-VBM
DMSG-VBM: Deep Multi-Source Graph-Enhanced Variational Bayesian Model for Predicting Higher-Order Microbe-Drug-Disease Associations

Exploring potential relationships among drugs, microbes, and diseases is critical for advancing disease prevention and drug development. Several computational models have been developed to predict high-order interactions in multi-omics biological systems. However, negative sampling-based methods risk generating false negatives, whereas tensor decomposition approaches typically require extensive hyperparameter tuning and exhibit limited flexibility when handling complex data architectures. To address these limitations, we propose DMSG-VBM, a deep multi-source graph-enhanced variational Bayesian model for drug-microbe-disease association prediction.  The model employs multi-source graph deep learning to generate prior expectations of latent variables and leverages an attention mechanism for adaptive fusion of multi-source features. Furthermore, DMSG-VBM integrates tensor-based Bayesian inference with multi-source graph deep learning within a probabilistic graphical model and adopts a variational expectation-maximization algorithm for adaptive inference of model parameters and latent variables. Experimental results on two benchmark datasets across four prediction scenarios demonstrate that DMSG-VBM achieves superior performance in triplet prediction compared to state-of-the-art methods, under both balanced and extremely imbalanced settings. It also attains higher hit rates in predictive tasks involving novel drugs, novel microbes, and novel diseases. Case studies further substantiate that DMSG-VBM effectively predicts potential associations among drugs, microbes, and diseases. The code and dataset are available at: https://github.com/Mayingjun20179/DMSG-VBM.

#The workflow of our proposed DMSG-VBM model

![image](https://github.com/Mayingjun20179/DMSG-VBM/blob/main/workflow.png)

#Environment Requirement

tensorly==0.8.1

torch==2.4.1+cu121

pandas==2.0.3

deepchem==2.8.0

rdkit==2022.9.4

networkx==2.8.8

torch-geometric==2.6.1

torch_scatter==2.1.2+pt24cu121

#Documentation

DATA1: Experimental data for baseline data Data1

DATA2: Experimental data for baseline data Data2

result1: After running the program, the location where the experimental result of the benchmark data Data1 is stored.

result2: After running the program, the location where the experimental result of the benchmark data Data2 is stored.

#Usage

First, install all the packages required by “requirements.txt”.

Second, run the program “Main_DMSGVBM_CV.py” to get all the prediction results of DMSG-VBM for the two benchmark datasets in the scenarios of CV_triplet, CV_drug, CV_micro and CV_dis.
