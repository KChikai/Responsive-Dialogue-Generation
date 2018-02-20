# coding: UTF-8
"""
テスト文に対してドメイン推定を行うスクリプト．
テストファイルは一行一文の形式で，出力ファイルは推定ドメインラベルとテキストのTAB区切りを出力する．
"""

import re
import io
import cPickle as pkl
import numpy as np
import theano
import theano.tensor as T
import batch_char as batch
import lasagne

from t2v import tweet2vec, load_params
from settings_char import N_BATCH, MAX_LENGTH, MAX_CLASSES, MODEL_PATH, TEST_INPUT, SAVE_PATH


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
    テキストファイルにタグを付与する関数
    :return:
    """
    # path
    model_path = MODEL_PATH
    text_path = TEST_INPUT
    save_path = SAVE_PATH


    # seq2seq用regex
    pattern = "(.+?)(\n|\r\n)"
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

    # Encoding cmnts
    cmnts = []
    Xt = []
    for line in io.open(text_path, 'r', encoding='utf-8'):
        m = r.search(line)
        if m is not None:
            cmnts.append(m.group(1))
            Xc_cmnt = m.group(1).replace(' ', '')       # 半角スペースの除去(tweet2vecに入力するため)
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
        for tag, cmnt in zip(out_pred, cmnts):
            f.write(tag + '\t' + cmnt + '\n')



if __name__ == '__main__':
    annotate_s2s_text()
