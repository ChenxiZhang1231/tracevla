#! /bin/bash
cd /inspire/hdd/global_user/zhangchenxi-253108310322/RLinf/
source switch_env openpi
bash examples/embodiment/run_embodiment.sh maniskill_hier_S5_prefix_xt_vt


source switch_env openpi
bash examples/embodiment/run_embodiment.sh maniskill_base_pi0


#现有方法使用Pi0分别输入xt,vt
source switch_env openpi
bash examples/embodiment/run_embodiment.sh maniskill_value_prefix_xt_vt_pi0


#现有方法训练Pi0
source switch_env openpi
bash examples/embodiment/run_embodiment.sh maniskill_hier_step50_pi05

#现有方法训练Pi0+grpo
source switch_env openpi
bash examples/embodiment/run_embodiment.sh maniskill_grpo_pi0