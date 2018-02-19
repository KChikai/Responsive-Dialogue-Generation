# -*- coding:utf-8 -*-
"""
fine tuning 用スクリプト
事前にトレーニングした seq2seq model が必要
"""

import os

os.environ["CHAINER_TYPE_CHECK"] = "0"

import pickle
import argparse
import numpy as np
import chainer
from chainer import cuda, optimizers, serializers
from util import JaConvCorpus
from pretrain_seq2seq import PreTrainSeq2Seq
from finetune_seq2seq import FineTuneSeq2Seq
from setting_param import EPOCH, FEATURE_NUM, HIDDEN_NUM, LABEL_NUM, LABEL_EMBED, BATCH_NUM, CORPUS_DIR, PRETRAIN_MODEL

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=EPOCH, type=int, help='number of epochs to learn')
parser.add_argument('--feature_num', '-f', default=FEATURE_NUM, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=HIDDEN_NUM, type=int, help='dimension of hidden layer')
parser.add_argument('--label_num', '-ln', default=LABEL_NUM, type=int, help='dimension of label layer')
parser.add_argument('--label_embed', '-le', default=LABEL_EMBED, type=int, help='dimension of label embed layer')
parser.add_argument('--batchsize', '-b', default=BATCH_NUM, type=int, help='learning minibatch size')
args = parser.parse_args()

# GPU settings
gpu_device = args.gpu
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = args.epoch
feature_num = args.feature_num
hidden_num = args.hidden_num
batchsize = args.batchsize
label_num = args.label_num
label_embed = args.label_embed


def remove_extra_padding(batch_list, reverse_flg=True):
    """
    remove extra padding
    :param batch_list: a list of a batch
    :param reverse_flg: whether a batch of sentences is reversed or not
    """
    remove_row = []
    # reverse order (padding first)
    if reverse_flg:
        for i in range(len(batch_list)):
            if sum(batch_list[i]) == -1 * len(batch_list[i]):
                remove_row.append(i)
            else:
                break
    # natural order (padding last)
    else:
        for i in range(len(batch_list))[::-1]:
            if sum(batch_list[i]) == -1 * len(batch_list[i]):
                remove_row.append(i)
            else:
                break
    return np.delete(batch_list, remove_row, axis=0)


def main():
    ###########################
    #### create dictionary ####
    ###########################

    if os.path.exists(CORPUS_DIR + 'dictionary.dict'):
        corpus = JaConvCorpus(create_flg=False, batch_size=batchsize, size_filter=True)
        corpus.load(load_dir=CORPUS_DIR)
    else:
        corpus = JaConvCorpus(create_flg=True, batch_size=batchsize, size_filter=True)
        corpus.save(save_dir=CORPUS_DIR)
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('Emotion size: ', len(corpus.emotion_set))

    # search word_threshold (general - emotional)
    ma = 0
    mi = 999999
    for word in corpus.emotion_set:
        wid = corpus.dic.token2id[word]
        if wid > ma:
            ma = wid
        if wid < mi:
            mi = wid
    word_threshold = mi

    ######################
    #### create model ####
    ######################

    # load pretrain model
    rough_model = PRETRAIN_MODEL
    pretrain_model = PreTrainSeq2Seq(all_vocab_size=len(corpus.dic.token2id), emotion_vocab_size=len(corpus.emotion_set),
                                     feature_num=feature_num, hidden_num=hidden_num, batch_size=batchsize,
                                     label_num=label_num, label_embed_num=label_embed, gpu_flg=args.gpu)
    serializers.load_hdf5(rough_model, pretrain_model)

    model = FineTuneSeq2Seq(all_vocab_size=len(corpus.dic.token2id), emotion_vocab_size=len(corpus.emotion_set),
                            feature_num=feature_num, hidden_num=hidden_num, batch_size=batchsize,
                            label_num=label_num, label_embed_num=label_embed, gpu_flg=args.gpu)

    # copy weights
    # encoder
    model.enc.xe.W = pretrain_model.enc.xe.W
    model.enc.eh.W = pretrain_model.enc.eh.W
    model.enc.hh.W = pretrain_model.enc.hh.W
    model.enc.eh_rev.W = pretrain_model.enc.eh_rev.W
    model.enc.hh_rev.W = pretrain_model.enc.hh_rev.W
    # ws
    model.ws.W = pretrain_model.ws.W
    # decoder
    model.dec.ye.W = pretrain_model.dec.ye.W
    model.dec.le.W = pretrain_model.dec.le.W
    model.dec.eh.W = pretrain_model.dec.eh.W          # label embed層が変わる場合コピーできない
    model.dec.hh.W = pretrain_model.dec.hh.W
    model.dec.vt.W = pretrain_model.dec.vt.W
    model.dec.wg.W = pretrain_model.dec.wg.W
    model.dec.we.W = pretrain_model.dec.we.W

    if args.gpu >= 0:
        model.to_gpu()
    optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.GradientClipping(5))
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    ##########################
    #### create ID corpus ####
    ##########################

    input_mat = []
    output_mat = []
    input_mat_rev = []
    label_mat = []
    topic_vec = []
    # label_index = [index for index in range(label_num)]
    max_input_ren = max_output_ren = 0
    print('start making corpus matrix...')
    for input_text, output_text in zip(corpus.fine_posts, corpus.fine_cmnts):

        # reverse an input and add eos tag
        output_text.append(corpus.dic.token2id["<eos>"])  # 出力の最後にeosを挿入

        # update max sentence length
        max_input_ren = max(max_input_ren, len(input_text))
        max_output_ren = max(max_output_ren, len(output_text))

        # make topic tag
        topic_label = output_text.pop(0)
        if topic_label == corpus.dic.token2id['__label__0']:
            topic_vec.append(0)
        elif topic_label == corpus.dic.token2id['__label__1']:
            topic_vec.append(1)
        else:
            print('no label error: ', topic_label)
            raise ValueError

        # make a list of lists
        input_mat.append(input_text)
        output_mat.append(output_text)

        # make emotion label lists TODO: 3値分類
        n_num = p_num = 0
        for word in output_text:
            if corpus.dic[word] in corpus.neg_words:
                n_num += 1
            if corpus.dic[word] in corpus.pos_words:
                p_num += 1
        if (n_num + p_num) == 0:
            label_mat.append([1 for _ in range(len(output_text))])
        elif n_num <= p_num:
            label_mat.append([2 for _ in range(len(output_text))])
        elif n_num > p_num:
            label_mat.append([0 for _ in range(len(output_text))])
        else:
            raise ValueError

    # convert topic label matrix to numpy.array
    topic_vec = np.array(topic_vec, dtype=np.int32)

    # make reverse corpus
    for input_text in input_mat:
        input_mat_rev.append(input_text[::-1])

    # padding (inputの文頭・outputの文末にパディングを挿入する)
    print('start labeling...')
    for li in input_mat:
        insert_num = max_input_ren - len(li)
        for _ in range(insert_num):
            li.append(corpus.dic.token2id['<pad>'])
    for li in output_mat:
        insert_num = max_output_ren - len(li)
        for _ in range(insert_num):
            li.append(corpus.dic.token2id['<pad>'])
    for li in input_mat_rev:
        insert_num = max_input_ren - len(li)
        for _ in range(insert_num):
            li.insert(0, corpus.dic.token2id['<pad>'])
    for li in label_mat:
        insert_num = max_output_ren - len(li)
        for _ in range(insert_num):
            li.append(corpus.dic.token2id['<pad>'])
    if len(output_mat) != len(label_mat):
        print('Output matrix and label matrix should have the same dimension.')
        raise ValueError

    # create batch matrix
    print('transpose...')
    input_mat = np.array(input_mat, dtype=np.int32).T
    input_mat_rev = np.array(input_mat_rev, dtype=np.int32).T
    output_mat = np.array(output_mat, dtype=np.int32).T
    label_mat = np.array(label_mat, dtype=np.int32).T

    # separate corpus into Train and Test TODO:実験時はテストデータとトレーニングデータに分離する
    print('split train and test...')
    train_input_mat = input_mat
    train_output_mat = output_mat
    train_input_mat_rev = input_mat_rev
    train_label_mat = label_mat
    train_topic_vec = topic_vec

    #############################
    #### train seq2seq model ####
    #############################

    accum_loss = 0
    train_loss_data = []
    print('start training...')
    for num, epoch in enumerate(range(n_epoch)):
        total_loss = 0
        batch_num = 0
        perm = np.random.permutation(len(corpus.fine_posts))

        # for training
        for i in range(0, len(corpus.fine_posts), batchsize):

            # select batch data
            input_batch = remove_extra_padding(train_input_mat[:, perm[i:i + batchsize]], reverse_flg=False)
            input_batch_rev = remove_extra_padding(train_input_mat_rev[:, perm[i:i + batchsize]], reverse_flg=True)
            output_batch = remove_extra_padding(train_output_mat[:, perm[i:i + batchsize]], reverse_flg=False)
            label_batch = remove_extra_padding(train_label_mat[:, perm[i:i + batchsize]], reverse_flg=False)
            topic_batch = train_topic_vec[perm[i:i + batchsize]]

            # Encode a sentence
            model.initialize(batch_size=input_batch.shape[1])  # initialize cell
            model.encode(input_batch, input_batch_rev, topic_label=topic_batch, train=True)

            # Decode from encoded context
            input_ids = xp.array([corpus.dic.token2id["<start>"] for _ in range(input_batch.shape[1])])
            for w_ids, l_ids in zip(output_batch, label_batch):
                loss, predict_mat = model.decode(input_ids, w_ids, label_id=l_ids, word_th=word_threshold, train=True)
                input_ids = w_ids
                accum_loss += loss

            # learn model
            model.cleargrads()      # initialize all grad to zero
            accum_loss.backward()   # back propagation
            optimizer.update()
            total_loss += float(accum_loss.data)
            batch_num += 1
            print('Epoch: ', num, 'Batch_num', batch_num, 'batch loss: {:.2f}'.format(float(accum_loss.data)))
            accum_loss = 0

        train_loss_data.append(float(total_loss / batch_num))

        # save model and optimizer
        if (epoch + 1) % 5 == 0:
            print('-----', epoch + 1, ' times -----')
            print('save the model and optimizer')
            serializers.save_hdf5('../data/seq2seq/' + str(epoch) + '_fine.model', model)
            serializers.save_hdf5('../data/seq2seq/' + str(epoch) + '_fine.state', optimizer)

    # save loss data
    with open('../data/seq2seq/loss_train_data.pkl', 'wb') as f:
        pickle.dump(train_loss_data, f)


if __name__ == "__main__":
    main()
