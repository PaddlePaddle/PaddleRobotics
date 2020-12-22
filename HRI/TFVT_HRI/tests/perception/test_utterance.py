import pytest
import codecs

from perception.utterance.eval import UtteranceEncoder
from perception.utterance.tokenizer import Tokenizer, WordpieceTokenizer
from perception.common.utils import cosine_sim

SIM_TEST_FILE = 'data/utterance_sim_test.txt'
BoW_MODEL = 'pretrain_models/simnet_bow_pairwise_pretrained_model'
BoW_VOCABULARY_PATH = 'pretrain_models/data/term2id.dict'

TOKENIZER_TEST_FILE = 'data/tokenizer/test.txt'

ERNIE_v1_VOCAB_PATH = 'pretrain_models/ERNIE_v1/vocab.txt'
ERNIE_v1_MODEL = 'pretrain_models/ERNIE_v1/params'
ERNIE_v1_CFG = 'pretrain_models/ERNIE_v1/ernie_config.json'


class TestUtteranceEncoder(object):
    def setup_class(self):
        self.pos_pairs, self.neg_pairs = [], []
        with codecs.open(SIM_TEST_FILE, 'r', 'utf-8') as f:
            for line in f.readlines():
                sentence_0, sentence_1, label = line.strip().split('\t')
                sentence_0 = sentence_0.split(' ')
                sentence_1 = sentence_1.split(' ')
                label = int(label)
                if label == 1:
                    self.pos_pairs.append((sentence_0, sentence_1))
                elif label == 0:
                    self.neg_pairs.append((sentence_0, sentence_1))

    def test_ernie_encoder(self):
        ernie_encoder = UtteranceEncoder(
            ERNIE_v1_MODEL, ERNIE_v1_VOCAB_PATH, algorithm='ernie_v1',
            model_cfg_path=ERNIE_v1_CFG)

        for sentence_0, sentence_1 in self.pos_pairs:
            sentence_0 = ''.join(sentence_0)
            sentence_1 = ''.join(sentence_1)
            encodings = ernie_encoder.get_encoding([sentence_0, sentence_1])
            sim = cosine_sim(encodings[0], encodings[1])
            print(sentence_0, sentence_1, sim)
            assert sim > 0.9

        for sentence_0, sentence_1 in self.neg_pairs:
            sentence_0 = ''.join(sentence_0)
            sentence_1 = ''.join(sentence_1)
            encodings = ernie_encoder.get_encoding([sentence_0, sentence_1])
            sim = cosine_sim(encodings[0], encodings[1])
            print(sentence_0, sentence_1, sim)
            assert sim < 0.9


class TestTokenizer(object):
    def setup_class(self):
        self.sentences = []
        with codecs.open(TOKENIZER_TEST_FILE, 'r', 'utf-8') as f:
            for line in f.readlines():
                self.sentences.append(line.strip())

        self.tokenizer = Tokenizer()

    def test_get_tokens(self):
        tokens_lst = self.tokenizer.get_tokens(self.sentences, with_tag=True)
        print(tokens_lst)
        assert len(tokens_lst) == len(self.sentences)


class TestWordpieceTokenizer(object):
    def setup_class(self):
        self.sentences = []
        with codecs.open(TOKENIZER_TEST_FILE, 'r', 'utf-8') as f:
            for line in f.readlines():
                self.sentences.append(line.strip())

        # append some English sentences
        en_sentences = [
            "I love to play football, but she doesn't."
        ]
        self.sentences.extend(en_sentences)
        self.tokenizer = WordpieceTokenizer(ERNIE_v1_VOCAB_PATH)

    def test_get_tokens(self):
        tokens_lst = self.tokenizer.get_tokens(self.sentences)
        print(tokens_lst)
        assert len(tokens_lst) == len(self.sentences)
