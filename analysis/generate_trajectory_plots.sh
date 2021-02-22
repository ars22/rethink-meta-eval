#! /bin/bash

output='fixS5_cifar-fs-base_r12_FOMAMLinnUpd5T20Vp0.01_n5s5q15tb4_Adam0.01det2040'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0'

export PYTHONPATH='..'

CUDA_VISIBLE_DEVICES="$device" python generate_trajectory_plots.py \
--n-runs 1 \
--fix-support 5 \
--fix-support-path ../runs/fixS5_cifar-fs-base_r12_FOMAMLinnUpd5T20Vp0.01_n5s5q15tb4_Adam0.01det2040/fixed_support_pool.pkl \
--preload-train True \
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
--dataset-path ../datasets/filelists/cifar-fs-base \
--img-side-len 32 \
--batch-size-val 4 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 2500 \
--eps 0. \
--output-folder ${output} \
--device cuda \
--device-number ${device} \
--checkpoint ../runs/fixS5_cifar-fs-base_r12_FOMAMLinnUpd5T20Vp0.01_n5s5q15tb4_Adam0.01det2040/ \
--log-interval 100
