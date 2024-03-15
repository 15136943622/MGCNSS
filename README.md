# MGCNSS
This repository provides a reference implementation of MGCNSS as described in the paper:
> MGCNSS: miRNA-Disease Association Prediction based on Multi-layer Graph Convolution and Distance-based Negative Sample Selection Strategy
>
> Chenguang Han
> 

Available at 

## Dependencies
Recent versions of the following packages for Python 3 are required:
* numpy==1.19.2
* torch==1.9.1+cu111
* scipy==1.5.4
* pandas==0.25.0

## Datasets：data
* disease: the four similarity matrices of disease
* mirna: the four similarity matrices of miRNA
* train: the train and test sets which contains 1:1, 1:5 and 1:10 data.
* Y: the initial feature obtained from RWR

## Usage
Run `Link_Prediction.py` directly to do MDA tasks.

## About `my_model.pkl`
* It is a trained model, you can do prediction directly by load this file through `torch.load('my_model.pkl')`.
* Then, please add those parameters to the model
*    'feature': the initial feature of nodes, size 878
*    'A': the fused similarity matrix of miRNA
*    'B': the fused similarity matrix of disease
*    'o_ass': miRNA-disease adjacent matrix
*    'layer': convolutional layer, you can set the value with 2
* Then, you can obtain the finial feature matrix of miRNAs and diseases, and you can do prediction tasks.
* The experimental parameter setting about this model: epoch-2000, learning rate-0.0005, embedding size-256, convolution layer-2.



