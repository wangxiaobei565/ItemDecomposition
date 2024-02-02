# Future Impact Decomposition in Request-level Recommendations

arxiv links:[Future Impact Decomposition in Request-level Recommendations](https://arxiv.org/abs/2401.16108)



## 0. Setup

```
conda create -n itema2c python=3.9
conda activate itema2c
conda install pytorch torchvision -c pytorch
conda install pandas matplotlib scikit-learn
pip install tqdm
conda install -c anaconda ipykernel
python -m ipykernel install --user --name weighta2c --display-name "ItemA2C"
```

## 1. Parpare for RL Simulator


We should set up dataset and  Response Model as Environment Component
#### 1. Data Preparation

For KuaiRand data preparation, run cells in KRData.ipynb. 

For ML1M data preparation, run cells in ML1MData.ipynb. 

#### 2. Pretrain User Response Model as Environment Component

Modify train_env.sh:
* Change the directories, data_path, and output_path for your dataset
* Set the following arguments with X in {ML1M, KuaiRand}:
  * --model {X}UserResponse\
  * --reader {X}DataReader\
  * --train_file ${data_path}{X}_b_train.csv\
  * --val_file ${data_path}{X}_b_test.csv\
* Set your model_path and log_path in the script.
* The env in our experiments is set by:
  *  kr_user_env_lr0.001_reg0.003_init
  *  ml1m_user_env_lr0.001_reg0.0001_final
 
```
conda activate itema2c
bash train_env.sh
```


## 2. Our code
We release our model itemA2C, itemA2C-W, itemA2C-M as itemA2C, itemA2C_W, itemA2C_model. The corresponding facade, policy, critic are concluded in the model file.
- itemA2C
- itemA2C_W
- itemA2C_model

## 3. Run code
#### Search optimal hyperparameter for different method(optinonal)
```
cd /code/scripts/hyperparameter_search
bash XXXX.sh
```
This step is to find optimal performance with adjustable hyperparameter. Our result can list as :
|dataset|method|actor_lr|critic_lr|weight_lr|
|--|--|--|--|--|
|ML1M|itemA2C|3e-4|3e-5|None|
|KuaiRand|itemA2C|3e-4|3e-5|None|
|ML1M|itemA2C-W|3e-4|3e-5|None|
|KuaiRand|itemA2C-W|3e-4|3e-5|None|
|ML1M|itemA2C-M|3e-5|3e-6|1e-8|
|KuaiRand|itemA2C-M|1e-3|3e-6|1e-7|


#### run the main experiments or ablation experiments
```
cd /code/scripts/XXXX
bash XXXX.sh
```

We give all of our training config in the scripts and the plot utils in the code to show the results visually.
#### Note:
- The experiments are easy to reproduce since our experiments are running in one GPU Tesla T4 with 15 GB memory.
- The log name must be right for User Response Model.
- The model save path can be changed by editing save_path and log_path after create the path.
