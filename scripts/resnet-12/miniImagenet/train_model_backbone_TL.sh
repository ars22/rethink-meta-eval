#! /bin/bash

output='TL_mini_r12_PNeuc_bs128_SGD0.1Drop90'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='1'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python main.py \
--fix-support 0 \
--model-type resnet_12 \
--avg-pool True \
--projection True \
--num-classes-train 64 \
--algorithm TransferLearning \
--scale-factor 10. \
--classifier-metric euclidean \
--classifier-type linear \
--dataset-path datasets/filelists/miniImagenet \
--img-side-len 84 \
--n-epochs 100 \
--batch-size-train 128 \
--val-frequency 3 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--do-one-shot-eval-too False \
--n-query-val 15 \
--n-iterations-val 5000 \
--query-aug True \
--preload-train True \
--optimizer-type SGDM \
--lr 0.1 \
--weight-decay 0.0005 \
--grad-clip 0. \
--drop-lr-epoch 90 \
--drop-factors 0.1 \
--lr-scheduler-type deterministic \
--eps 0. \
--restart-iter 0 \
--output-folder ${output} \
--device-number ${device} \
--log-interval 100 > logs/${output}_train.log &
tail -f logs/${output}_train.log