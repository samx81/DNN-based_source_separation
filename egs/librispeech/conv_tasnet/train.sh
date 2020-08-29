#!/bin/bash

. ./path.sh

exp_dir="$1"
continue_from="$2"

n_sources=2

wav_root="../../../dataset/LibriSpeech"
train_json_path="../../../dataset/LibriSpeech/train-clean-100/train-100-${n_sources}mix.json"
valid_json_path="../../../dataset/LibriSpeech/dev-clean/valid-${n_sources}mix.json"

sr=16000

# Encoder & decoder
enc_basis='trainable'
dec_basis='trainable'
enc_nonlinear='relu' # window_fn is activated if enc_basis='trainable'
window_fn='hamming' # window_fn is activated if enc_basis='Fourier' or dec_basis='Fourier'
N=64
L=16

# Separator
H=256
B=128
Sc=128
P=3
X=6
R=3
dilated=1
separable=1
causal=0
sep_nonlinear='prelu'
sep_norm=1
mask_nonlinear='sigmoid'

# Criterion
criterion='sisdr'

# Optimizer
optimizer='adam'
lr=0.001
weight_decay=0
max_norm=5

batch_size=4
epochs=100

use_cuda=1
overwrite=0
seed=111

tag=""

if [ ${enc_nonlinear} = 'trainable' ]; then
    tag="${tag}enc-${enc_nonlinear}_"
fi

if [ ${enc_basis} = 'Fourier' -o ${dec_basis} = 'Fourier' ]; then
    tag="${tag}${window_fn}-window_"
fi

save_dir="${exp_dir}/${n_sources}mix/${enc_basis}-${dec_basis}/${criterion}/N${N}_L${L}_B${B}_H${H}_Sc${Sc}_P${P}_X${X}_R${R}/${tag}dilated${dilated}_separable${separable}_causal${causal}_${sep_nonlinear}_norm${sep_norm}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"


model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"
log_dir="${save_dir}/log"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

train.py \
--wav_root ${wav_root} \
--train_json_path ${train_json_path} \
--valid_json_path ${valid_json_path} \
--sr ${sr} \
--enc_basis ${enc_basis} \
--dec_basis ${dec_basis} \
--enc_nonlinear ${enc_nonlinear} \
--window_fn ${window_fn} \
-N ${N} \
-L ${L} \
-B ${B} \
-H ${H} \
-Sc ${Sc} \
-P ${P} \
-X ${X} \
-R ${R} \
--dilated ${dilated} \
--separable ${separable} \
--causal ${causal} \
--sep_nonlinear ${sep_nonlinear} \
--sep_norm ${sep_norm} \
--mask_nonlinear ${mask_nonlinear} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--optimizer ${optimizer} \
--lr ${lr} \
--weight_decay ${weight_decay} \
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