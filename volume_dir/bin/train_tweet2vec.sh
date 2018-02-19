#!/bin/bash

#########################
# train tweet2vec model #
#########################

# decide local env
cd ../tweet2vec
pyenv local anaconda-2.4.0

# choose data directory
exp="tweet2vec"
train_data="../data/$exp/tweet2vec_topic_trainer.txt"
val_data="../data/$exp/tweet2vec_topic_tester.txt"
model_path="../data/$exp/"

# train tweet2vec model
echo "Training tweet2vec ..."
THEANO_FLAGS=device=gpu,floatX=float32 python char.py ${train_data} ${val_data} ${model_path}
