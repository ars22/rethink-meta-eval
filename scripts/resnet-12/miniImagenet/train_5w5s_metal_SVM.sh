#! /bin/bash

output='metal_mini_r12_SVM_n5s15q6tb8_SGD0.1Drop20'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='1'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python main.py \
--fix-support 0 \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm SVM \
--scale-factor 7. \
--classifier-metric euclidean \
--dataset-path datasets/filelists/miniImagenet \
--img-side-len 84 \
--n-epochs 60 \
--batch-size-train 8 \
--n-way-train 5 \
--n-shot-train 15 \
--n-query-train 6 \
--n-iters-per-epoch 1000 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--do-one-shot-eval-too True \
--n-query-val 15 \
--n-iterations-val 5000 \
--support-aug True \
--query-aug True \
--randomize-query False \
--preload-train True \
--optimizer-type SGDM \
--lr 0.1 \
--weight-decay 0.0005 \
--grad-clip 0. \
--drop-lr-epoch 20,40,50 \
--lr-scheduler-type deterministic \
--eps 0. \
--restart-iter 0 \
--output-folder ${output} \
--device cuda \
--device-number ${device} \
--log-interval 100 > logs/${output}_train.log &
tail -f logs/${output}_train.log