#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sr=${sr_k}000
duration=4
max_or_min='min'

split_type="cv_sub"
wav_root="data/voicebank/${split_type}"
test_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tt_mix"

# Encoder & decoder
enc_bases='trainable' # choose from 'trainable','Fourier', or 'trainableFourier'
dec_bases='trainable' # choose from 'trainable','Fourier', 'trainableFourier', or 'pinv'
enc_nonlinear='relu' # enc_nonlinear is activated if enc_bases='trainable' and dec_bases!='pinv'
window_fn='' # window_fn is activated if enc_bases='Fourier' or dec_bases='Fourier'
D=64
M=16 # M corresponds to the window length (samples) in this script.

# Separator
H=128
K=100
P=50
Q=32
N=6
J=8
sep_norm=1
sep_nonlinear='relu'
sep_dropout=1e-1
mask_nonlinear='relu'
causal=0

# Criterion
criterion='sisdr'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=1e-6
max_norm=5 # 0 is handled as no clipping

batch_size=4
epochs=100

use_cuda=1
overwrite=0
seed=111
gpu_id="0"


. ./path.sh
. parse_options.sh || exit 1

prefix=""

if [ ${enc_bases} = 'trainable' -a -n "${enc_nonlinear}" -a ${dec_bases} != 'pinv' ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

if [ ${enc_bases} = 'Fourier' -o ${dec_bases} = 'Fourier' ]; then
    prefix="${preffix}${window_fn}-window_"
fi

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${duration}sec/${enc_bases}-${dec_bases}/${criterion}/D${D}_M${M}_H${H}_K${K}_P${P}_Q${Q}_H${H}_N${N}_J${J}/${prefix}causal${causal}_norm${sep_norm}_drop${sep_dropout}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

model_choice="best"

model_dir="${save_dir}/model"
model_path="${model_dir}/${model_choice}.pth"
log_dir="${save_dir}/log"
out_dir="${save_dir}/test/${split_type}"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

test.py \
--test_wav_root ${wav_root} \
--test_list_path ${test_list_path} \
--sr ${sr} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
