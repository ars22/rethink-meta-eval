#! /bin/bash

output='metal_MI_r12_PNeuc_n5s5q15Vtb4_SGD0.1Drop204050_basetest_chkpt_060_metal_MI_r12_PNeuc_n5s5q15Vtb4_SGD0.1Drop204050_basetest_chkpt_035'
# method_dataset_model_innerAlg_config_outerOpt_misc
device='0'

export PYTHONPATH='..'

CUDA_VISIBLE_DEVICES="$device" python compute_novel_acc_variance.py \
--sample 0 \
--n-chosen-classes 20 \
--n-runs 100 \
--preload-train False \
--model-type resnet_12 \
--avg-pool True \
--projection False \
--classifier-type no-classifier \
--num-classes-train 0 \
--algorithm-1 ProtoNet \
--algorithm-2 ProtoNet \
--scale-factor 10. \
--classifier-metric euclidean \
--dataset-path ../datasets/filelists/miniImagenet \
--img-side-len 84 \
--batch-size-val 4 \
--n-way-val 5 \
--n-shot-val 5 \
--n-query-val 15 \
--n-iterations-val 2500 \
--eps 0. \
--output-folder ${output} \
--device cuda \
--device-number ${device} \
--checkpoint-1 ../runs/metal_MI_r12_PNeuc_n5s5q15Vtb4_SGD0.1Drop204050_basetest/chkpt_060.pt \
--checkpoint-2 ../runs/metal_MI_r12_PNeuc_n5s5q15Vtb4_SGD0.1Drop204050_basetest/chkpt_035.pt \
--log-interval 100
