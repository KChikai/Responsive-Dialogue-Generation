#!/bin/bash

#######################
# train seq2seq model #
#######################

# decide local env
cd ../seq2seq
pyenv local anaconda3-4.2.0

# pretrain seq2seq model
gpu_num=1
echo "PreTraining seq2seq model ..."
python train_rough.py --gpu ${gpu_num}

# fine-tune seq2seq model
echo "Fine-Tuning seq2seq model ..."
python train_fine.py --gpu ${gpu_num}
