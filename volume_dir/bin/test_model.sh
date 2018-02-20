#!/bin/bash

# volume_dir/data/test_input.txt から出力文を生成するテストスクリプト
# tweet2vec の出力ファイル名は volume_dir/tweet2vec/setting_char.py に定義している

# decide local env
cd ../tweet2vec
pyenv local anaconda-2.4.0

# predict domain label by tweet2vec
python predict.py

# generate outputs by seq2seq
cd ../seq2seq
python interpreter.py