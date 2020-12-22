# coding: utf8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unicodedata
import numpy as np
from paddle import fluid

from perception.common.utils import load_vocab


class Tokenizer(object):
    """
    Sentences tokenizer.
    Current version only supports Chinese.
    """
    def __init__(self):
        this_dir = os.path.realpath(os.path.dirname(__file__))
        tokenizer_dir = os.path.realpath(
            os.path.join(this_dir, '../../data/tokenizer'))
        model_dir = os.path.join(tokenizer_dir, 'model')
        tag_file = os.path.join(tokenizer_dir, 'tag.dic')
        zh_word_dict = os.path.join(tokenizer_dir, 'word.dic')

        # NOTE: this is a dict for special char replacement
        zh_rep_dict = os.path.join(tokenizer_dir, 'q2b.dic')

        self.word2id_dict = load_vocab(
            zh_word_dict, with_unk0=False, word_first=False, value_to_int=True)
        self.rep_dict = load_vocab(
            zh_rep_dict, with_unk0=False, word_first=True, value_to_int=False)
        self.tag_dict = load_vocab(
            tag_file, with_unk0=False, word_first=True, value_to_int=False)

        self.place = fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)
        self.infer_scope = fluid.Scope()
        self._build_infer_program(model_dir)

    def get_tokens(self, sentences, with_tag=False):
        """
        For each sentence, return a list of splitted words, with
        optional tag paired with the splitted word in a tuple.

        Args:
            sentences (list of str): list of Chinese sentences.
        """
        seqs, seqs_lens = [], []
        for sentence in sentences:
            seq = []
            for w in sentence:
                if w in self.rep_dict:
                    w = self.rep_dict[w]
                if w in self.word2id_dict:
                    seq.append(self.word2id_dict[w])
                else:
                    seq.append(self.word2id_dict['OOV'])
            seqs.append(seq)
            seqs_lens.append(len(seq))

        with fluid.scope_guard(self.infer_scope):
            seqs_lod = fluid.create_lod_tensor(seqs, [seqs_lens], self.place)
            crf_decode = self.exe.run(self.infer_program,
                                      feed={'word': seqs_lod},
                                      fetch_list=self.fetch_targets,
                                      return_numpy=False)[0]
            crf_ndarray = np.array(crf_decode)
            lod_info = crf_decode.lod()[0]

        tokens_lst = []
        for sent_idx in range(len(sentences)):
            tokens, cur_token, word_idx, tag_ = [], [], 0, ''
            for tag_id in range(lod_info[sent_idx], lod_info[sent_idx+1]):
                # For the meaning of tags, see https://git.io/JezI6
                tag = self.tag_dict[str(crf_ndarray[tag_id][0])]
                # NOTE: assume every word in sentences is also in dict
                # so it has corresponding id in seqs
                word = sentences[sent_idx][word_idx]
                if tag.endswith('-B') or tag.endswith('O'):
                    if len(cur_token) > 0:
                        if with_tag:
                            token = (''.join(cur_token), tag_)
                        else:
                            token = ''.join(cur_token)
                        tokens.append(token)
                    cur_token = [word]
                else:
                    cur_token.append(word)

                tag_ = tag.split('-')[0]
                word_idx += 1

            if len(cur_token) > 0:
                if with_tag:
                    token = (''.join(cur_token), tag_)
                else:
                    token = ''.join(cur_token)
                tokens.append(token)

            tokens_lst.append(tokens)
        return tokens_lst

    def _build_infer_program(self, model_dir):
        with fluid.scope_guard(self.infer_scope):
            self.infer_program, self.feed_target_names, self.fetch_targets = \
                fluid.io.load_inference_model(model_dir, self.exe)


class WordpieceTokenizer(object):
    """
    Wordpiece tokenizer for English and Chinese sentences.

    NOTE:
    Wordpiece tokenizer differs from conventional tokenizer to tokenize
    word to substring, instead of tokenizing sentences to groups of words.
    For example,
    'I love to play football':
    wordpiece tokenize => ['I', 'love', 'to', 'play', 'foot', '##ball']
    conventional tokenize => ['I' (r), 'love' (v), 'to' (p), 'play' (v),
    'football' (n)]

    '我爱踢足球。' => ['我', '爱', '踢', '足', '球', '。']
    TODO: check connecting tokens '##' for Chinese, '球' or '##球',
    for current version, use '球'
    """
    def __init__(self,
                 vocab_file,
                 unk_token='[UNK]',
                 use_lower_case=True,
                 max_input_chars_per_word=100):
        self.vocab = load_vocab(vocab_file, with_unk0=False, word_first=True)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token
        self.use_lower_case = use_lower_case
        self.max_input_chars_per_word = max_input_chars_per_word

    def get_tokens(self, sentences):
        """
        For each sentence, return a list of wordpieces.

        Args:
            sentences (list of str): list of sentences.
        """
        sentences = [self._clean_text(s) for s in sentences]
        sentences = [self._insert_whitespace_for_zh(s) for s in sentences]
        # some words are connecting with punctuation, e.g. ['football.']
        original_tokens_lst = [s.strip().split(' ') for s in sentences]

        if self.use_lower_case:
            original_tokens_lst = [[t.lower() for t in tokens]
                                   for tokens in original_tokens_lst]

        # split words punctuations and symbols, e.g. ['football', '.']
        word_level_tokens_lst = [self._split_punc_and_symbols(tokens)
                                 for tokens in original_tokens_lst]
        wordpiece_level_tokens_lst = [self._greedy_match_wordpiece(tokens)
                                      for tokens in word_level_tokens_lst]
        return wordpiece_level_tokens_lst

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens if token in self.vocab]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab[idx] for idx in ids if idx in self.inv_vocab]

    def _clean_text(self, text):
        """
        Remove invalid chars, unicode control chars and marks
        """
        output = []
        for char in text:
            cat = unicodedata.category(char)
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or \
               cat.startswith('M') or \
               (char not in ['\t', '\n', '\r'] and cat.startswith('C')):
                # Invalid unicode or control chars, see
                # https://www.fileformat.info/info/unicode/category/index.htm
                continue

            if char in [' ', '\t', '\n', '\r'] or cat == 'Zs':
                # treat separater as whitespace
                output.append(' ')
            else:
                output.append(char)

        return ''.join(output)

    def _insert_whitespace_for_zh(self, text):
        output = []
        for char in text:
            if self._is_zh_char(char):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)

        return ''.join(output)

    def _is_zh_char(self, char):
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        cp = ord(char)
        if (cp >= 0x4E00 and cp <= 0x9FFF) or \
           (cp >= 0x3400 and cp <= 0x4DBF) or \
           (cp >= 0x20000 and cp <= 0x2A6DF) or \
           (cp >= 0x2A700 and cp <= 0x2B73F) or \
           (cp >= 0x2B740 and cp <= 0x2B81F) or \
           (cp >= 0x2B820 and cp <= 0x2CEAF) or \
           (cp >= 0xF900 and cp <= 0xFAFF) or \
           (cp >= 0x2F800 and cp <= 0x2FA1F):
            return True

        return False

    def _split_punc_and_symbols(self, tokens):
        new_tokens = []
        for token in tokens:
            chars = list(token)
            idx, split = 0, []
            while idx < len(chars):
                cat = unicodedata.category(chars[idx])
                if cat.startswith('P') or cat.startswith('S'):
                    if len(split) > 0:
                        new_tokens.append(''.join(split))
                        split = []
                    new_tokens.append(chars[idx])
                else:
                    split.append(chars[idx])

                idx += 1

            if len(split) > 0:
                new_tokens.append(''.join(split))

        return new_tokens

    def _greedy_match_wordpiece(self, tokens):
        """
        Try to match wordpieces following greedy longest match.
        """
        new_tokens = []
        for token in tokens:
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                new_tokens.append(self.unk_token)
                continue

            start, is_bad, wordpieces = 0, False, []
            while start < len(chars):
                end, wordpiece = len(chars), ''
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr

                    if substr in self.vocab:
                        wordpiece = substr
                        break

                    end -= 1
                if wordpiece == '':
                    # Not match any wordpiece from current `start`
                    is_bad = True
                    break

                wordpieces.append(wordpiece)
                start = end

            if is_bad:
                new_tokens.append(self.unk_token)
            else:
                new_tokens.extend(wordpieces)

        return new_tokens
