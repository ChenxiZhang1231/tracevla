#! /bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export HYDRA_FULL_ERROR=1

# Accept command line arguments or use defaults
CKPT_PATH=${1:-null}                          # 第1个参数：检查点路径
TOTAL_NUM_ENVS=${2:-320}                      # 第2个参数：环境数量
EVAL_ROLLOUT_EPOCH=${3:-1}                    # 第3个参数：评估轮数
EVAL_NAME=${4:-four_tasks_eval}               # 第4个参数：实验名称

CONFIG_NAME="maniskill_ppo_pi05_four_tasks"

for env_id in \
    "PutCarrotOnPlateInScene-v1" \
    "PutSpoonOnTableClothInScene-v1" \
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1" \
    "PutEggplantInBasketScene-v1"; \
do
    # Eggplant 任务需要更多步数
    if [ "$env_id" = "PutEggplantInBasketScene-v1" ]; then
        max_steps=120
    else
        max_steps=60
    fi

    LOG_DIR="${REPO_PATH}/logs/eval/${EVAL_NAME}/$(date +'%Y%m%d-%H:%M:%S')-${env_id}"
    MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
    mkdir -p "${LOG_DIR}"

    CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ \
        --config-name ${CONFIG_NAME} \
        runner.logger.log_path=${LOG_DIR} \
        algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH} \
        env.eval.total_num_envs=${TOTAL_NUM_ENVS} \
        env.eval.max_episode_steps=${max_steps} \
        env.eval.max_steps_per_rollout_epoch=${max_steps} \
        env.eval.init_params.id=${env_id} \
        env.eval.init_params.max_episode_steps=${max_steps} \
        runner.ckpt_path=${CKPT_PATH}"

    echo ${CMD} > ${MEGA_LOG_FILE}
    ${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
done
