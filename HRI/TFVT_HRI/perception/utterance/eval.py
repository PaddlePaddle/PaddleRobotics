import os
from paddle import fluid

from perception.common.utils import load_vocab
from perception.utterance.tokenizer import WordpieceTokenizer
from perception.utterance.bow import BoW
from perception.utterance.ernie_v1 import ErnieConfig, ErnieModel, \
    convert_example_to_record, pad_batch_records


class UtteranceEncoder(object):
    def __init__(self,
                 model_dir,
                 vocabulary_path,
                 gpu=None,
                 algorithm='bow',
                 model_cfg_path=None):
        if not os.path.exists(model_dir):
            raise ValueError('The model path `%s` does not exit.' % model_dir)
        if not os.path.exists(vocabulary_path):
            raise ValueError('The vocabulary path `%s` does not exit.' %
                             vocabulary_path)
        if algorithm not in ['bow', 'ernie_v1']:
            raise NotImplementedError

        if gpu is not None and type(gpu) is int:
            self.place = fluid.CUDAPlace(gpu)
        else:
            self.place = fluid.CPUPlace()

        # TODO: unify the different usage style of vocabulary
        # for BoW and ERNIE. The first uses raw dict, while the later
        # uses vocab dict in tokenizer
        if algorithm == 'bow':
            self.vocabulary = load_vocab(
                vocabulary_path, with_unk0=True, word_first=True,
                value_to_int=True)
        elif algorithm == 'ernie_v1':
            self.ernie_config = ErnieConfig(model_cfg_path)
            self.wordpiece_tokenizer = WordpieceTokenizer(vocabulary_path)

        self.algorithm = algorithm
        self.exe = fluid.Executor(self.place)
        self.infer_scope = fluid.Scope()

        with fluid.scope_guard(self.infer_scope):
            self._build_infer_program(model_dir)

    def get_encoding(self, sentences):
        """
        Get encoding vector for given sentences.

        Args:
            sentences (list of sentence): list of sentences.
                When algorithm is 'bow', each sentence is a list of tokens.
                When algorithm is 'ernie_v1', each sentence is a string.
        """
        if self.algorithm == 'bow':
            feed = self._process_bow_feed(sentences)
        elif self.algorithm == 'ernie_v1':
            feed = self._process_ernie_feed(sentences)

        with fluid.scope_guard(self.infer_scope):
            encoding = self.exe.run(
                self.infer_program,
                feed=feed,
                fetch_list=self.fetch_list)[0]

        return encoding

    def _build_infer_program(self, model_dir):
        main, startup = fluid.Program(), fluid.Program()
        with fluid.program_guard(main, startup):
            if self.algorithm == 'bow':
                net = BoW(len(self.vocabulary))
            elif self.algorithm == 'ernie_v1':
                net = ErnieModel(self.ernie_config)

            self.infer_program, self.fetch_list = net.infer(main)
            fluid.io.load_persistables(
                self.exe, model_dir, main_program=self.infer_program)

    def _process_bow_feed(self, sentences):
        seqs, seqs_lens = [], []
        for sentence in sentences:
            seq = [self.vocabulary[token] for token in sentence
                   if token in self.vocabulary]
            seqs.append(seq)
            seqs_lens.append(len(seq))

        with fluid.scope_guard(self.infer_scope):
            seqs_lod = fluid.create_lod_tensor(seqs, [seqs_lens], self.place)
            return {'seq': seqs_lod}

    def _process_ernie_feed(self, sentences):
        pad_id = self.wordpiece_tokenizer.vocab['[PAD]']
        records = [convert_example_to_record(self.wordpiece_tokenizer, s)
                   for s in sentences]
        src_ids, sent_ids, pos_ids, input_mask = pad_batch_records(
            records, pad_id)

        feed = {
            'src_ids': src_ids,
            'sent_ids': sent_ids,
            'pos_ids': pos_ids,
            'input_mask': input_mask
        }
        return feed
