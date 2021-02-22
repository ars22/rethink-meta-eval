#! /bin/bash

output='fixS5_FC100-base_r12_PN_n20s5q15tb1_SGD0.1Drop20_BNFix'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python main.py \
--fix-support 5 \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path datasets/filelists/FC100-base \
--img-side-len 32 \
--n-epochs 15 \
--batch-size-train 1 \
--n-way-train 20 \
--n-shot-train 5 \
--n-query-train 15 \
--n-iters-per-epoch 1000 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--do-one-shot-eval-too False \
--n-query-val 15 \
--n-iterations-val 5000 \
--support-aug False \
--query-aug False \
--randomize-query True \
--preload-train True \
--optimizer-type SGDM \
--lr 0.1 \
--weight-decay 0.0005 \
--grad-clip 0. \
--drop-lr-epoch 5 \
--lr-scheduler-type deterministic \
--eps 0. \
--restart-iter 0 \
--output-folder ${output} \
--device cuda \
--device-number ${device} \
--log-interval 100 > logs/${output}_train.log &
tail -f logs/${output}_train.log