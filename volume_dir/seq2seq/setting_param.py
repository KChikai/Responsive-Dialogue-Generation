# seq2seq parameters

# data to train seq2seq
CORPUS_DIR = '../data/seq2seq/corpus/'
W2V_MODEL_PATH = '../data/neo_model.vec'
PRETRAIN_DATA = '../data/rough_pair_corpus.txt'
FINETUNE_DATA = '../data/fine_pair_corpus.txt'
NEG_DATA = '../data/neg-extend.txt'
POS_DATA = '../data/pos-extend.txt'

PRETRAIN_MODEL = '../data/seq2seq/19_rough.model'
TEST_MODEL = '../data/seq2seq/19_fine.model'

T2V_OUTPUT = '../data/t2v_output.txt'

# corpus size
MAX_LENGTH = 25
WORD_NUM = 40000

# epoch
EPOCH = 20

# batch size
BATCH_NUM = 200

# dimension for encoder and decoder
FEATURE_NUM = 256
HIDDEN_NUM = 512

# dropout rate
DROP_OUT = 0.2

# dimension for emotion label
LABEL_NUM = 3
LABEL_EMBED = 64

# dimension for topic label
TOPIC_NUM = 2
TOPIC_EMBED = 64


