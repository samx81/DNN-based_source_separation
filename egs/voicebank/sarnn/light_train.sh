#!/bin/bash

exp_dir="./exp_test"
continue_from=""
tag=""

n_sources=2
sr_k=16 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sr=${sr_k}000
duration=4
valid_duration=10
max_or_min='min'

train_wav_root="data/voicebank/tr_sub_sub"
valid_wav_root="data/voicebank/cv"

train_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tr_mix"
valid_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_cv_mix"

# Encoder & decoder
basis="trainable"
enc_basis='trainable' # choose from 'trainable','Fourier', or 'trainableFourier'
dec_basis='trainable' # choose from 'trainable','Fourier', 'trainableFourier', or 'pinv'
enc_nonlinear='relu' # enc_nonlinear is activated if enc_basis='trainable' and dec_basis!='pinv'
window_fn='' # window_fn is activated if enc_basis='Fourier' or dec_basis='Fourier'

echo $basis

N=64
L=16 # L corresponds to the window length (samples) in this script.

# Separator
F=64
H=128
K=100
P=50
B=2
d_ff=256
h=4
causal=0
sep_norm=1
sep_nonlinear='relu'
sep_dropout=0
mask_nonlinear='relu'

# Criterion
criterion='sisnr'

# Optimizer
optimizer='adam'
k1=2e-1
k2=4e-4
warmup_steps=4000
weight_decay=0
max_norm=5 # 0 is handled as no clipping

batch_size=1
epochs=100

use_cuda=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

enc_basis=$basis 
dec_basis=$basis

echo $basis
echo $enc_basis

prefix=""

if [ ${enc_basis} = 'trainable' -a -n "${enc_nonlinear}" -a ${dec_basis} != 'pinv' ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

if [ ${enc_basis} = 'Fourier' -o ${dec_basis} = 'Fourier' ]; then
    prefix="${preffix}${window_fn}-window_"
fi

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${duration}sec/${enc_basis}-${dec_basis}/${criterion}/N${N}_L${L}_F${F}_H${H}_K${K}_P${P}_B${B}_d-ff${d_ff}_h${h}/${prefix}causal${causal}_norm${sep_norm}_${sep_nonlinear}_drop${sep_dropout}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-k1${k1}-k2${k2}-decay${weight_decay}-warmup${warmup_steps}_clip${max_norm}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"
log_dir="${save_dir}/log"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

train.py \
--train_wav_root ${train_wav_root} \
--valid_wav_root ${valid_wav_root} \
--train_list_path ${train_list_path} \
--valid_list_path ${valid_list_path} \
--sr ${sr} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--enc_basis ${enc_basis} \
--dec_basis ${dec_basis} \
--enc_nonlinear "${enc_nonlinear}" \
--window_fn "${window_fn}" \
-N ${N} \
-L ${L} \
-F ${F} \
-K ${K} \
-P ${P} \
-B ${B} \
-d_ff ${d_ff} \
--sep_num_heads ${h} \
--causal ${causal} \
--sep_norm ${sep_norm} \
--sep_nonlinear ${sep_nonlinear} \
--sep_dropout ${sep_dropout} \
--mask_nonlinear ${mask_nonlinear} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--optimizer ${optimizer} \
--k1 ${k1} \
--k2 ${k2} \
--weight_decay ${weight_decay} \
--warmup_steps ${warmup_steps} \
--max_norm ${max_norm} \
--batch_size ${batch_size} \
--epochs ${epochs} \
--model_dir "${model_dir}" \
--loss_dir "${loss_dir}" \
--sample_dir "${sample_dir}" \
--continue_from "${continue_from}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/train_${time_stamp}.log"
