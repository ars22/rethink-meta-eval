#! /bin/bash

output='fixS5_MI_r12_PNeuc_n5s5q15Vtb4_SGD0.1Drop204050_basetest'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python eval.py \
--fix-support 5 \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path datasets/filelists/miniImagenet \
--img-side-len 84 \
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
--checkpoint runs/fixS5_MI_r12_PNeuc_n5s5q15Vtb4_SGD0.1Drop204050_basetest/chkpt_053.pt \
--log-interval 100 > logs/${output}_eval.log &
tail -f logs/${output}_eval.log