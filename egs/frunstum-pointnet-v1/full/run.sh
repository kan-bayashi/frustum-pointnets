#!/bin/bash

. ./path.sh
. ./cmd.sh

stage=012
model=frustum_pointnets_v1
train=./conf/train.txt
valid=./conf/val.txt
rgb_valid=./conf/rgb_detection_val.txt
ratio=1.0

num_point=1024
max_epoch=201
batch_size=32
decay_step=800000
decay_rate=0.5
tag=

. parse_options.sh
set -e

# STAGE 0 {{{
if echo ${stage} | grep -q 0; then
    echo "###########################################################"
    echo "#                 DATA PREPARATION STEP                   #"
    echo "###########################################################"
    echo "Start data prepartion."
    if [ ! -e data/${ratio} ]; then
        mkdir -p data/${ratio}
    fi
    n_all=$(wc -l ${train} | awk '{print $1}')
    n_subset=$(echo "${n_all} * ${ratio}" | bc)
    n_subset=${n_subset%.*}
    sort -R ${train} | head -n "${n_subset}" | sort > data/${ratio}/train.txt
    cp ${valid} data/${ratio}/val.txt
    cp ${rgb_valid} data/${ratio}/rgb_detection_val.txt
    echo "number of training data = $(wc -l ${train})"
    echo "number of validation data = $(wc -l ${valid})"
    train=data/${ratio}/train.txt
    valid=data/${ratio}/val.txt
    rgb_valid=data/${ratio}/rgb_detection_val.txt
    ${train_cmd} exp/prepare_data/prepare_data_ratio${ratio}.log \
        prepare_data.py \
            --gen_train \
            --gen_val \
            --gen_val_rgb_detection \
            --train ${train} \
            --valid ${valid} \
            --rgb_valid ${rgb_valid} \
            --write_dir dump/${ratio}
    echo "Successfully finished data preparation."
else
    train=data/${ratio}/train.txt
    valid=data/${ratio}/val.txt
    rgb_valid=data/${ratio}/rgb_detection_val.txt
fi
# }}}


# STAGE 1 {{{
if [ ! -n "${tag}" ];then
    expdir=exp/tr_${model}_r${ratio}_np${num_point}_bs${batch_size}_ds$(( decay_step / 1000 ))k_dr${decay_rate}
else
    expdir=exp/tr_${tag}
fi
if echo ${stage} | grep -q 1; then
    echo "###########################################################"
    echo "#                     TRAINING STEP                       #"
    echo "###########################################################"
    echo "Start training."
    ${cuda_cmd} "${expdir}/train.log" \
        train.py \
            --gpu 0 \
            --model ${model} \
            --train dump/${ratio}/frustum_carpedcyc_train.pickle \
            --valid dump/${ratio}/frustum_carpedcyc_val.pickle \
            --log_dir "${expdir}" \
            --num_point ${num_point} \
            --max_epoch ${max_epoch} \
            --batch_size ${batch_size} \
            --decay_step ${decay_step} \
            --decay_rate ${decay_rate}
    echo "Successfully finished training."
fi
# }}}


# STAGE 2 {{{
if echo ${stage} | grep -q 2; then
    ${cuda_cmd} "${expdir}/test.log" \
        test.py \
            --gpu 0 \
            --num_point ${num_point} \
            --model ${model} \
            --model_path "${expdir}/model.ckpt" \
            --output "${expdir}/results_val_from_rgb_detection" \
            --data_path dump/${ratio}/frustum_carpedcyc_val_rgb_detection.pickle \
            --idx_path ${valid} \
            --from_rgb_detection

    # kitti native evaluation
    ${train_cmd} "${expdir}/results_val_from_rgb_detection/kitti_eval.log" \
        evaluate_object_3d_offline \
            "$PRJ_ROOT/dataset/KITTI/object/training/label_2" \
            "${expdir}/results_val_from_rgb_detection"
fi
# }}}
