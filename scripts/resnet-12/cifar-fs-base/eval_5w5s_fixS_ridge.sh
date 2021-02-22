#! /bin/bash

output='fixS15_cifar-fs-base_r12_Ridge_n5s15q6tb8_SGD0.1Drop204050'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='1'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python eval.py \
--eot-model True \
--fix-support 15 \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 0 \
--algorithm Ridge \
--scale-factor 7. \
--classifier-metric euclidean \
--dataset-path datasets/filelists/cifar-fs-base \
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
--checkpoint runs/fixS15_cifar-fs-base_r12_Ridge_n5s15q6tb8_SGD0.1Drop204050/chkpt_060.pt \
--log-interval 100 > logs/${output}_evaleot.log &
tail -f logs/${output}_evaleot.log
