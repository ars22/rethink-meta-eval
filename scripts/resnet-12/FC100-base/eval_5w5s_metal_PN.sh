#! /bin/bash

output='metal_FC100_r12_PNeuc_n20s5q15Vtb1_SGD0.1det05_basetest'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='3'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python eval.py \
--fix-support 0 \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path datasets/filelists/FC100-base \
--img-side-len 32 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--do-one-shot-eval-too False \
--n-query-val 15 \
--n-iterations-val 5000 \
--eps 0. \
--output-folder ${output} \
--device cuda \
--device-number ${device} \
--checkpoint runs/metal_FC100_r12_PNeuc_n20s5q15Vtb1_SGD0.1det05_basetest/chkpt_007.pt \
--log-interval 100 > logs/${output}_eval.log &
tail -f logs/${output}_eval.log