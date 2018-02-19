# coding: UTF-8
"""
データのアノテート用スクリプト
"""

import re
import io
import cPickle as pkl
import numpy as np
import theano
import theano.tensor as T
import batch_char as batch
import lasagne
from scipy import spatial

from t2v import tweet2vec, load_params
from settings_char import N_BATCH, MAX_LENGTH, MAX_CLASSES


#################
## for predict ##
#################

def invert(d):
    out = {}
    for k, v in d.iteritems():
        out[v] = k
    return out


def classify(tweet, t_mask, params, n_classes, n_chars):
    emb_layer = tweet2vec(tweet, t_mask, params, n_chars)
    l_dense = lasagne.layers.DenseLayer(emb_layer, n_classes, W=params['W_cl'], b=params['b_cl'],
                                        nonlinearity=lasagne.nonlinearities.softmax)
    return lasagne.layers.get_output(l_dense), lasagne.layers.get_output(emb_layer)


def annotate_s2s_text():
    """
    seq2seqに食わせるデータ形式のテキストファイルのコメント側にのみ
    タグを付与する関数
    :return:
    """
    # path
    model_path = 'model/tweet2vec/'                 # 使用するモデル
    text_path = '../data/pair_corpus.txt'           # 入力データ
    save_path = '../data/pair_corpus_emotion.txt'   # 出力データ

    # seq2seq用regex
    pattern = "(.+?)(\t)(.+?)(\n|\r\n)"
    r = re.compile(pattern)

    print("Loading model params...")
    params = load_params('%s/best_model.npz' % model_path)

    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)
    inverse_labeldict = invert(labeldict)

    print("Building network...")
    tweet = T.itensor3()
    t_mask = T.fmatrix()
    predictions, embeddings = classify(tweet, t_mask, params, n_classes, n_char)

    print("Compiling theano functions...")
    predict = theano.function([tweet, t_mask], predictions)
    encode = theano.function([tweet, t_mask], embeddings)

    # Encoding cmnts
    posts = []
    cmnts = []
    Xt = []
    for line in io.open(text_path, 'r', encoding='utf-8'):
        m = r.search(line)
        if m is not None:
            posts.append(m.group(1))
            cmnts.append(m.group(3))
            Xc_cmnt = m.group(3).replace(' ', '')       # 半角スペースの除去(tweet2vecに入力するため)
            Xt.append(Xc_cmnt[:MAX_LENGTH])

    out_pred = []
    numbatches = len(Xt) / N_BATCH + 1
    print 'number of batches', numbatches
    for i in range(numbatches):
        xr = Xt[N_BATCH*i:N_BATCH*(i+1)]
        x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
        p = predict(x, x_m)
        ranks = np.argsort(p)[:, ::-1]

        for idx, item in enumerate(xr):
            out_pred.append([inverse_labeldict[r] if r in inverse_labeldict else 'UNK' for r in ranks[idx, :5]][0])

        print i, 'batches end...'

    # Save result
    with io.open(save_path, 'w') as f:
        for post, tag, cmnt in zip(posts, out_pred, cmnts):
            f.write(post + '\t' + tag + ' ' + cmnt + '\n')


def annotate_s2s_emo_text():
    """
    seq2seqに食わせるデータ形式のテキストファイルのコメント側にのみ
    タグを付与する関数
    :return:
    """
    # path
    model_path = 'model/tweet2vec/'                 # 使用するモデル
    text_path = '../data/pair_corpus.txt'           # 入力データ
    pos_save_path = '../data/pair_corpus_pos_emotion.txt'   # 出力データ
    neg_save_path = '../data/pair_corpus_neg_emotion.txt'  # 出力データ
    neu_save_path = '../data/pair_corpus_neu_emotion.txt'  # 出力データ

    # seq2seq用regex
    pattern = "(.+?)(\t)(.+?)(\n|\r\n)"
    r = re.compile(pattern)

    print("Loading model params...")
    params = load_params('%s/best_model.npz' % model_path)

    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)
    inverse_labeldict = invert(labeldict)

    print("Building network...")
    tweet = T.itensor3()
    t_mask = T.fmatrix()
    predictions, embeddings = classify(tweet, t_mask, params, n_classes, n_char)

    print("Compiling theano functions...")
    predict = theano.function([tweet, t_mask], predictions)
    encode = theano.function([tweet, t_mask], embeddings)

    # Encoding cmnts
    posts = []
    cmnts = []
    Xt = []
    for line in io.open(text_path, 'r', encoding='utf-8'):
        m = r.search(line)
        if m is not None:
            posts.append(m.group(1))
            cmnts.append(m.group(3))
            Xc_cmnt = m.group(3).replace(' ', '')       # 半角スペースの除去(tweet2vecに入力するため)
            Xt.append(Xc_cmnt[:MAX_LENGTH])

    out_pred = []
    numbatches = len(Xt) / N_BATCH + 1
    print 'number of batches', numbatches

    with io.open(pos_save_path, 'w') as f_pos:
        with io.open(neg_save_path, 'w') as f_neg:
            with io.open(neu_save_path, 'w') as f_neu:
                for i in range(numbatches):
                    xr = Xt[N_BATCH*i:N_BATCH*(i+1)]
                    x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
                    p = predict(x, x_m)
                    ranks = np.argsort(p)[:, ::-1]

                    for idx, item in enumerate(xr):
                        top_1_index = ranks[idx, :1]
                        if p[idx, top_1_index] > 0.95:
                            out_pred.append([inverse_labeldict[r] if r in inverse_labeldict else 'UNK' for r in ranks[idx, :1]][0])
                        else:
                            out_pred.append('neutral')

                    print i, 'batches end...'

                    # Save result
                    for post, tag, cmnt in zip(posts[N_BATCH*i:N_BATCH*(i+1)], out_pred, cmnts[N_BATCH*i:N_BATCH*(i+1)]):
                        if tag == 'happy':
                            f_pos.write(post + '\t' + tag + ' ' + cmnt + '\n')
                        elif tag == 'angry':
                            f_neg.write(post + '\t' + tag + ' ' + cmnt + '\n')
                        else:
                            f_neu.write(post + '\t' + tag + ' ' + cmnt + '\n')


def make_embeddings():
    """
    embeddingを作成する関数
    :return:
    """
    # path
    model_path = 'model/tweet2vec/'                 # 使用するモデル
    text_path = '../data/emotion.txt'               # 入力データ
    save_path = '../data/embedding_emodata.npy'     # 出力データ

    # seq2seq用regex
    pattern = "(.+?)(\t)(.+?)(\n|\r\n)"
    r = re.compile(pattern)

    print("Loading model params...")
    params = load_params('%s/best_model.npz' % model_path)

    print("Loading dictionaries...")
    with open('%s/dict.pkl' % model_path, 'rb') as f:
        chardict = pkl.load(f)
    with open('%s/label_dict.pkl' % model_path, 'rb') as f:
        labeldict = pkl.load(f)
    n_char = len(chardict.keys()) + 1
    n_classes = min(len(labeldict.keys()) + 1, MAX_CLASSES)

    print("Building network...")
    tweet = T.itensor3()
    t_mask = T.fmatrix()
    predictions, embeddings = classify(tweet, t_mask, params, n_classes, n_char)

    print("Compiling theano functions...")
    predict = theano.function([tweet, t_mask], predictions)
    encode = theano.function([tweet, t_mask], embeddings)

    # Encoding cmnts
    posts = []
    cmnts = []
    Xt = []
    out_emb = []
    for line in io.open(text_path, 'r', encoding='utf-8'):
        m = r.search(line)
        if m is not None:
            posts.append(m.group(1))
            cmnts.append(m.group(3))
            Xc_cmnt = m.group(3).replace(' ', '')       # 半角スペースの除去(tweet2vecに入力するため)
            Xt.append(Xc_cmnt[:MAX_LENGTH])

    numbatches = len(Xt) / N_BATCH + 1
    print 'number of batches', numbatches
    for i in range(numbatches):
        xr = Xt[N_BATCH*i:N_BATCH*(i+1)]
        x, x_m = batch.prepare_data(xr, chardict, n_chars=n_char)
        e = encode(x, x_m)

        for idx, item in enumerate(xr):
            out_emb.append(e[idx, :])

        print i, 'batch end...'

    with open(save_path, 'w') as f:
        np.save(f, np.asarray(out_emb))


def save_text():

    emo_data_path = '../data/emotion_bin/tweet2vec_trainer.txt'
    cmnt_data_path = '../data/pair_corpus.txt'
    save_emo_path = '../data/emo.pkl'
    save_post_path = '../data/post.pkl'
    save_cmnt_path = '../data/cmnt.pkl'

    # seq2seq用regex
    pattern = "(.+?)(\t)(.+?)(\n|\r\n)"
    r = re.compile(pattern)

    emos = []
    posts = []
    cmnts = []
    for line in io.open(emo_data_path, 'r', encoding='utf-8'):
        m = r.search(line)
        if m is not None:
            emos.append(m.group(3))
    for line in io.open(cmnt_data_path, 'r', encoding='utf-8'):
        m = r.search(line)
        if m is not None:
            posts.append(m.group(1))
            cmnts.append(m.group(3))

    with open(save_emo_path, 'wb') as f:
        pkl.dump(emos, f)
    with open(save_post_path, 'wb') as f:
        pkl.dump(posts, f)
    with open(save_cmnt_path, 'wb') as f:
        pkl.dump(cmnts, f)


def similarity():
    emo_path = '../data/embedding_emodata.npy'
    cmnt_path = '../data/embedding_cmnts.npy'
    save_emo_path = '../data/emo.pkl'
    save_post_path = '../data/cmnt.pkl'
    save_cmnt_path = '../data/cmnt.pkl'
    save_path = '../data/emo_pair_corpus.txt'

    with io.open(emo_path, 'rb') as f:
        emo_embed = np.load(f)
    with io.open(cmnt_path, 'rb') as f:
        cmnt_embed = np.load(f)
    with open(save_emo_path, 'rb') as f:
        emos_text = pkl.load(f)
    with open(save_post_path, 'rb') as f:
        posts_text = pkl.load(f)
    with open(save_cmnt_path, 'rb') as f:
        cmnts_text = pkl.load(f)

    save_index = []
    for i in range(cmnt_embed.shape[0]):
        sims = [(k, spatial.distance.cosine(cmnt_embed[i], emo_embed[k])) for k in range(emo_embed.shape[0])
                if (1 - spatial.distance.cosine(cmnt_embed[i], emo_embed[k])) > 0.9]
        sorted_sims = sorted(sims, key=lambda sim: sim[1])
        if len(sorted_sims) != 0:
            sim = sorted_sims[0]
            if sim[0] < (emo_embed.shape[0] / 2):
                save_index.append(('angry', i))
            elif sim[0] > (emo_embed.shape[0] / 2):
                save_index.append(('happy', i))
            else:
                print(i, sim)
                raise ValueError

        # print cmnts_text[i]
        # for sim in sorted_sims[:5]:
        #     print (1 - sim[1]), emos_text[sim[0]]

        if (i + 1) % 100 == 0:
            print i + 1, 'end'

    with io.open(save_path, 'w') as f:
        for tag, index in save_index:
            f.write(posts_text[i] + '\t' + tag + ' ' + cmnts_text[i] + '\n')


if __name__ == '__main__':
    # annotate_s2s_text()
    # annotate_s2s_emo_text()
    # make_embeddings()
    # save_text()
    similarity()