# seq2seq parameters

# corpus size
MAX_LENGTH = 25
WORD_NUM = 40000

# epoch
EPOCH = 100

# data to train seq2seq
DATA_DIR = '../data/seq2seq/'
CORPUS_DIR = '../data/seq2seq/corpus/'

W2V_MODEL_PATH = DATA_DIR + 'neo_model.vec'
PRETRAIN_DATA = DATA_DIR + 'rough_pair_corpus.txt'
FINETUNE_DATA = DATA_DIR + 'fine_pair_corpus.txt'
NEG_DATA = DATA_DIR + 'neg-extend.txt'
POS_DATA = DATA_DIR + 'pos-extend.txt'

PRETRAIN_MODEL = DATA_DIR + str(EPOCH-1) + '_rough.model'
TEST_MODEL = DATA_DIR + str(EPOCH-1) + '_fine.model'

T2V_OUTPUT = '../data/t2v_output.txt'

LOG_PATH = DATA_DIR + 'log.txt'

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


