#! /bin/bash

output='metal_MI_r12_PNeuc_n64s5q5Vtb1_SGD0.1det'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python eval.py \
--preload-train False \
--eot-model False \
--fix-support 0 \
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
--checkpoint runs/metal_MI_r12_PNeuc_n64s5q5Vtb1_SGD0.1det/chkpt_024.pt \
--log-interval 100 > logs/${output}_eval.log &
tail -f logs/${output}_eval.log