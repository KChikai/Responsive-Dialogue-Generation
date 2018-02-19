# -*- coding:utf-8 -*-

"""
topic label + emotion label を入力に加えた時の出力の確認用スクリプト．
tweet2vec を除いたモデル
入力文の後に半角＋数字（トピック）＋半角＋数字（感情）を入力することでラベルを挿入する．
（注）入力文中には半角スペースを入力しない．

ex)
こんにちは，今日もいい天気ですね！ 0 2

"""

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import argparse
import unicodedata
from chainer import serializers, cuda
from util import JaConvCorpus
from finetune_seq2seq import FineTuneSeq2Seq
from setting_param import FEATURE_NUM, HIDDEN_NUM, LABEL_NUM, LABEL_EMBED, TOPIC_NUM, CORPUS_DIR, TEST_MODEL

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--feature_num', '-f', default=FEATURE_NUM, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=HIDDEN_NUM, type=int, help='dimension of hidden layer')
parser.add_argument('--label_num', '-ln', default=LABEL_NUM, type=int, help='dimension of label layer')
parser.add_argument('--label_embed', '-le', default=LABEL_EMBED, type=int, help='dimension of label embed layer')
parser.add_argument('--topic_num', '-tn', default=TOPIC_NUM, type=int, help='dimension of topic layer')
parser.add_argument('--bar', '-b', default='0', type=int, help='whether to show the graph of loss values or not')
parser.add_argument('--lang', '-l', default='ja', type=str, help='the choice of a language (Japanese "ja" or English "en" )')
parser.add_argument('--beam_search', '-be', default=True, type=bool, help='show results using beam search')
args = parser.parse_args()

# GPU settings
gpu_device = args.gpu
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()


def parse_ja_text(text):
    """
    Function to parse Japanese text.
    :param text: string: sentence written by Japanese
    :return: list: parsed text
    """
    import MeCab
    mecab = MeCab.Tagger("mecabrc")
    mecab.parse('')

    # list up noun
    mecab_result = mecab.parseToNode(text)
    parse_list = []
    while mecab_result is not None:
        if mecab_result.surface != "":
            parse_list.append(unicodedata.normalize('NFKC', mecab_result.surface).lower())
        mecab_result = mecab_result.next

    return parse_list


def interpreter(data_path, model_path):
    """
    Run this function, if you want to talk to seq2seq model.
    if you type "exit", finish to talk.
    :param data_path: the path of corpus you made model learn
    :param model_path: the path of model you made learn
    :return:
    """
    # call dictionary class
    corpus = JaConvCorpus(create_flg=False)
    corpus.load(load_dir=data_path)
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = FineTuneSeq2Seq(all_vocab_size=len(corpus.dic.token2id), emotion_vocab_size=len(corpus.emotion_set),
                            feature_num=args.feature_num, hidden_num=args.hidden_num,
                            label_num=args.label_num, label_embed_num=args.label_embed, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)
    emo_label_index = [index for index in range(args.label_num)]
    topic_label_index = [index for index in range(args.topic_num)]

    # run conversation system
    print('The system is ready to run, please talk to me!')
    print('( If you want to end a talk, please type "exit". )')
    print('')
    while True:
        print('>> ', end='')
        sentence = input()
        if sentence == 'exit':
            print('See you again!')
            break

        # check a sentiment tag
        input_vocab = sentence.split(' ')
        emo_label_id = input_vocab.pop(-1)
        topic_label_id = input_vocab.pop(-1)
        label_false_flg = 1

        for index in emo_label_index:
            if emo_label_id == str(index):
                emo_label_id = index               # TODO: ラベルのインデックスに注意．今は3値分類 (0, 1, 2)
                label_false_flg = 0
                break
        if label_false_flg:
            print('caution: you donot set any enable tags! (emotion label)')
            emo_label_id = -1

        # check a topic tag                        # TODO: 本当はユーザ側の指定ではなく，tweet2vecの判定から決定する
        label_false_flg = 1
        for index in topic_label_index:
            if topic_label_id == str(index):
                topic_label_id = index             # TODO: ラベルのインデックスに注意．今は3値分類 (0, 1, 2)
                label_false_flg = 0
                break
        if label_false_flg:
            print('caution: you donot set any enable tags! (topic label)')
            topic_label_id = -1

        input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in parse_ja_text(sentence)]
        input_vocab_rev = input_vocab[::-1]

        # convert word into ID
        input_sentence = [corpus.dic.token2id[word] for word in input_vocab if not corpus.dic.token2id.get(word) is None]
        input_sentence_rev = [corpus.dic.token2id[word] for word in input_vocab_rev if not corpus.dic.token2id.get(word) is None]

        model.initialize(batch_size=1)
        if args.beam_search:
            hypotheses = model.beam_search(model.initial_state_function, model.generate_function,
                                           input_sentence, input_sentence_rev, start_id=corpus.dic.token2id['<start>'],
                                           end_id=corpus.dic.token2id['<eos>'], emo_label_id=emo_label_id,
                                           topic_label_id=topic_label_id)
            for hypothesis in hypotheses:
                generated_indices = hypothesis.to_sequence_of_values()
                generated_tokens = [corpus.dic[i] for i in generated_indices]
                print("--> ", " ".join(generated_tokens))
        else:
            sentence = model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 20,
                                      emo_label_id=emo_label_id, topic_label_id=topic_label_id,
                                      word2id=corpus.dic.token2id, id2word=corpus.dic)
        print("-> ", sentence)
        print('')


if __name__ == '__main__':
    interpreter(CORPUS_DIR, TEST_MODEL)
