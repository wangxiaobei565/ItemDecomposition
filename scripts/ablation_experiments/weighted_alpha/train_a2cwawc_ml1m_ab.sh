mkdir -p output

# ml1m environment

mkdir -p output/ml1m/
mkdir -p output/ml1m/env/
mkdir -p output/ml1m/env/log/
mkdir -p output/ml1m/agents/alpha/
mkdir -p output/ml1m/agents/alpha/wawc_alpha/

output_path="output/ml1m/"
log_name="ml1m_user_env_lr0.001_reg0.0001_final"


N_ITER=30000
CONTINUE_ITER=0
GAMMA=0.9
TOPK=1
EMPTY=0

MAX_STEP=20
INITEP=0.01
REG=0.00003
NOISE=0.1
ELBOW=0.1
EP_BS=32
BS=64
SEED=3
SCORER="WideDeep"
CRITIC_LR=0.001
ACTOR_LR=0.0001
BEHAVE_LR=0
TEMPER_RATE=1.0
ALPHA=0.5

# for MAX_STEP in 20
for ALPHA_A in 0 0.2 0.4 0.6 0.8 0.999 1.2 
do
    for ALPHA_C in 0 0.2 0.4 0.6 0.8 0.999 1.2 
    do
        for SCORER in "WideDeep"        
        do
            for SEED in 11 13 17 19 23
            do
                for ACTOR_LR in 0.00003
                do
                    mkdir -p ${output_path}agents/alpha/wawc_alpha//A2CWAWC_ml1m_alpha_a${ALPHA_A}_alpha_c${ALPHA_C}_seed${SEED}/

                    python train_ac.py\
                        --env_class ML1MEnvironment_GPU\
                        --policy_class OneStagePolicy_with_${SCORER}\
                        --critic_class QCritic\
                        --agent_class itemA2C_WWC\
                        --facade_class OneStageFacade_TD\
                        --seed ${SEED}\
                        --cuda 0\
                        --env_path ${output_path}env/${log_name}.env\
                        --max_step_per_episode ${MAX_STEP}\
                        --initial_temper ${MAX_STEP}\
                        --reward_func mean_with_cost\
                        --urm_log_path ${output_path}env/log/${log_name}.model.log\
                        --state_encoder_feature_dim 32\
                        --state_encoder_attn_n_head 4\
                        --state_encoder_hidden_dims 128\
                        --policy_actionnet_hidden 64\
                        --critic_hidden_dims 256 64\
                        --slate_size 6\
                        --buffer_size 100000\
                        --start_timestamp 2000\
                        --noise_var ${NOISE}\
                        --empty_start_rate ${EMPTY}\
                        --save_path ${output_path}agents/alpha/wawc_alpha/A2CWAWC_ml1m_alpha_a${ALPHA_A}_alpha_c${ALPHA_C}_seed${SEED}/model\
                        --episode_batch_size ${EP_BS}\
                        --batch_size ${BS}\
                        --alpha_a ${ALPHA_A}\
                        --alpha_c ${ALPHA_C}\
                        --actor_lr ${ACTOR_LR}\
                        --critic_lr ${CRITIC_LR}\
                        --behavior_lr ${BEHAVE_LR}\
                        --actor_decay ${REG}\
                        --critic_decay ${REG}\
                        --behavior_decay ${REG}\
                        --target_mitigate_coef 0.01\
                        --gamma ${GAMMA}\
                        --n_iter ${N_ITER}\
                        --initial_greedy_epsilon ${INITEP}\
                        --final_greedy_epsilon ${INITEP}\
                        --elbow_greedy ${ELBOW}\
                        --check_episode 10\
                        --topk_rate ${TOPK}\
                        > ${output_path}agents/alpha/wawc_alpha/A2CWAWC_ml1m_alpha_a${ALPHA_A}_alpha_c${ALPHA_C}_seed${SEED}/log
                done
            done
        done
    done
done
