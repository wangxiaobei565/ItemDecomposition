# Future Impact Decomposition in Request-level Recommendations



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


## 2. Our code
We release our model itemA2C, itemA2C-W, itemA2C-M as itemA2C, itemA2C_W, itemA2C_model. The corresponding facade, policy, critic are concluded in the model file.
- itemA2C
- itemA2C_W
- itemA2C_model

## 3. Run code
#### Search optimal hyperparameter for different method(optinonal)
```
cd MARLRec/code/scripts/hyperparameter_search
bash XXXX.sh
```
This step is to find optimal performance with adjustable hyperparameter. Our result can list as :
|dataset|method|actor_lr|critic_lr|Supervised_lr|
|--|--|--|--|--|
|ML1M|req_HAC|3e-5|1e-3|None|
|KuaiRand|req_HAC|3e-5|3e-4|None|
|ML1M|req_DDPG|1e-5|3e-3|None|
|KuaiRand|req_DDPG|3e-5|3e-2|None|
|ML1M|req_Supervision|None|None|0.0001|
|KuaiRand|req_Supervision|None|None|0.0001|
|ML1M|item_SlateQ|None|1e-4|None|
|KuaiRand|item_SlateQ|None|1e-3|None|
|ML1M|req_A2C|3e-5|3e-4|None|
|KuaiRand|req_A2C|3e-4|3e-3|None|
|ML1M|item_A2C|3e-4|3e-5|None|
|KuaiRand|item_A2C|3e-4|3e-5|None|
|ML1M|weighted actor|3e-4|3e-5|None|
|KuaiRand|weighted actor|3e-4|3e-5|None|
|ML1M|weighted critic|3e-4|3e-5|None|
|KuaiRand|weighted critic|3e-4|3e-5|None|
|ML1M|weighted both|3e-4|3e-5|None|
|KuaiRand|weighted both|3e-4|3e-5|None|

#### run the main experiments or ablation experiments
```
cd MARLRec/code/scripts/XXXX
bash XXXX.sh
```

We give all of our training config in the scripts and the plot utils in the code to show the results visually.
#### Note:
- The experiments are easy to reproduce since our experiments are running in one GPU Tesla T4 with 15 GB memory.
- The log name must be right for User Response Model.
- The model save path can be changed by editing save_path and log_path after create the path.
