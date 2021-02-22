#! /bin/bash

output='metal_MI_r12_PNeuc_n5s5q15Vtb4_SGD0.1Drop204050_basetest_chkpt_060'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0'

export PYTHONPATH='..'

CUDA_VISIBLE_DEVICES="$device" python compute_base_acc_variance.py \
--n-examples-per-class 100 \
--n-runs 100 \
--preload-train True \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--classifier-type no-classifier \
--num-classes-train 0 \
--algorithm ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path ../datasets/filelists/miniImagenet \
--img-side-len 84 \
--batch-size-val 2 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 5000 \
--eps 0. \
--output-folder ${output} \
--device cuda \
--device-number ${device} \
--checkpoint ../runs/metal_MI_r12_PNeuc_n5s5q15Vtb4_SGD0.1Drop204050_basetest/chkpt_060.pt \
--log-interval 100
