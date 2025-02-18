<h2 align="center">
 Leveraging 3D Gaussian for Temporal Knowledge Graph Embedding
</h2>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
</p>


<p align="center">
Codes for the paper Codes for the paper Leveraging 3D Gaussian for Temporal Knowledge Graph Embedding
</p>


## 


### Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name tkge_env python=3.10
source activate tkge_env
conda install --file requirements.txt -c pytorch
```


### Datasets

```
python process_icews.py

python process_gdelt.py
```

This will create the files required to compute the filtered metrics.

### Reproducing results of 3DG-TE

In order to reproduce the results of 3DG-TE on the four datasets in the paper,  run the following commands

```
CUDA_VISIBLE_DEVICES=0  python  learner.py --dataset ICEWS14    --emb_reg 0.0045 --time_reg 1e-2  --rank 1500 

CUDA_VISIBLE_DEVICES=0  python  learner.py --dataset ICEWS05-15 --emb_reg 0.002  --time_reg 0.1   --rank 1600

CUDA_VISIBLE_DEVICES=0  python  learner.py --dataset GDELT      --emb_reg 0.001  --time_reg 0.001 --rank 1500
```



### Evaluate

We provide a evaluate file under `best_checkpoints` folder, and you could evaluate 3DG-TE by running the following command: 

```
# (Here is an example with the GDELT dataset.)
CUDA_VISIBLE_DEVICES=0  python -u  test.py --dataset GDELT  --rank 1500 --batch_size 2000 
```




