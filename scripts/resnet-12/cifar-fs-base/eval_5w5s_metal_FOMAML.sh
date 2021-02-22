#! /bin/bash

output='metal_cifar-fs-base_r12_FOMAMLinnUpd5T20Vp0.01_n5s5q15tb4_Adam0.005det204050'
# method_dataset_model_innerAlg_config_outerOpt 
device='1'
mkdir -p logs
mkdir -p runs

CUDA_VISIBLE_DEVICES="$device" nohup python eval.py \
--fix-support 0 \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--num-classes-train 5 \
--classifier-type linear \
--algorithm InitBasedAlgorithm \
--init-meta-algorithm FOMAML \
--inner-update-method sgd \
--alpha 0.01 \
--num-updates-inner-train 5 \
--num-updates-inner-val 20 \
--classifier-metric euclidean \
--dataset-path datasets/filelists/cifar-fs-base \
--img-side-len 32 \
--batch-size-val 1 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 5000 \
--eps 0. \
--output-folder ${output} \
--device cuda \
--device-number ${device} \
--checkpoint runs/metal_cifar-fs-base_r12_FOMAMLinnUpd5T20Vp0.01_n5s5q15tb4_Adam0.005det204050/chkpt_045.pt \
--log-interval 100 > logs/${output}_eval.log &
tail -f logs/${output}_eval.log