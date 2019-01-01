import json
import logging
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import string
import csv
from utils import tokenize_preprocess_japanese_sent, google_translate, bing_translate, tokenize_preprocess_english_sent, tokenize_preprocess_japanese_sent, normalize_tokenized_sent, normalize_tokenized_answers
from overrides import overrides
import os
import ntpath

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import Field, TextField, IndexField, MetadataField

from white_space_word_tokenizer import WhiteSpaceWordTokenizer

from utils import create_sent_idx_dic
# Import NMT module.
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
from nmt.test import trans_from_files_beam, trans_from_files

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

seed = 3

@DatasetReader.register("squad_ml")
class SQuADReaderML(DatasetReader):
    """
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``Nine``)
        This class has been implemented for multilingual Question Ansering task,
        the input paragrphs and questins has already been separated by white spaces, and we do
        not need to have additional tokenier.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self, tokenizer=None, token_indexers=None, lazy=None):
        super().__init__(lazy)
        self._tokenizer = WhiteSpaceWordTokenizer()
        self._token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        # Add model file paths and train resources.

    # TODO: Only use the v5 path!
    def set_nmt_models_resources(self, trans_embedding_model, trans_encdec_model,
                                 trans_train_source, trans_train_target,
                                 use_question_tag=True, replace_UNK=False,
                                 version=4, online_trans=True, beam_search=False, soft=False):
        self.trans_embedding_model = trans_embedding_model
        self.trans_encdec_model = trans_encdec_model
        self.trans_train_source = trans_train_source
        self.trans_train_target = trans_train_target
        self.use_question_tag = use_question_tag
        self.replace_UNK = replace_UNK
        self.version = version
        self.online_trans = online_trans
        self.beam_search = beam_search
        self.soft = soft

    def set_squad_test_resources_ja(self):
        self.lang = "Ja"

        if not os.path.exists("trans_result/ja/v5"):
            os.makedirs("trans_result/ja/v5")
        self.question_file_path = \
            "../data/ja_question_v5.csv"
        self.source_context_file_path = \
            "../data/ja_question_v5_context.csv"
        trans_result_dir_name = \
            os.path.join("trans_result/ja/v5",
                         ntpath.basename(self.trans_embedding_model).split(".bin")[0])

        if not os.path.exists(trans_result_dir_name):
            os.makedirs(trans_result_dir_name)
        self.question_trans_file_path = os.path.join(
            trans_result_dir_name, "TRANS.question.txt.new")
        self.context_trans_file_path = os.path.join(
            trans_result_dir_name, "TRANS.txt.new")
        self.context_attention_file_path = os.path.join(
            trans_result_dir_name, "ATTN.txt.new")

        self.japanese_question_file_path = '../nmt/japanese_question_tmp.txt'
        self.japanese_context_file_path = '../nmt/japanese_context_tmp.txt'

    def set_squad_test_resources_fr(self):
        self.lang = "Fr"
        if not os.path.exists("trans_result/fr"):
            os.makedirs("trans_result/fr")

        if not os.path.exists("trans_result/fr/v5"):
            os.makedirs("trans_result/fr/v5")
        self.question_file_path = \
            "../data/fr_question_v5.csv"
        self.source_context_file_path = \
            "../data/fr_question_v5_context.csv"
        trans_result_dir_name = \
            os.path.join("trans_result/fr/v5",
                         ntpath.basename(self.trans_embedding_model).split(".bin")[0])

        if not os.path.exists(trans_result_dir_name):
            os.makedirs(trans_result_dir_name)
        self.question_trans_file_path = os.path.join(
            trans_result_dir_name, "TRANS.question.txt.new")
        self.context_trans_file_path = os.path.join(
            trans_result_dir_name, "TRANS.txt.new")
        self.context_attention_file_path = os.path.join(
            trans_result_dir_name, "ATTN.txt.new")

        self.japanese_question_file_path = '../nmt/french_question_tmp.txt'
        self.japanese_context_file_path = '../nmt/french_context_tmp.txt'

    def set_google_translate_mode(self, use_google_translate):
        self.use_google_translate = use_google_translate

    def set_bing_translate_mode(self, use_bing_translate):
        self.use_bing_translate = use_bing_translate

    @overrides
    # TODO: Refactor _read() func.
    def _read(self, question_file_path):
        sent_idx_dic = create_sent_idx_dic(self.source_context_file_path)

        if self.online_trans:
            # 1. Create tmp japanese question files.
            ja_q_tmp = open(self.japanese_question_file_path, "w")

            with open(self.question_file_path,  newline='') as f:
                dataReader = csv.reader(f)
                header = next(dataReader)
                for row in dataReader:
                    if self.lang == "Ja":
                        question = tokenize_preprocess_japanese_sent(row[3])
                        if self.use_question_tag == True:
                            ja_q_tmp.write(question + " <QS>\n")
                        else:
                            ja_q_tmp.write(question + "\n")
                    elif self.lang == "Fr":
                        ja_q_tmp.write(row[3].lower() + "\n")

            ja_q_tmp.close()
            if self.lang == "Fr":
                normalize_tokenized_sent(
                    self.japanese_question_file_path, self.lang)

            # 2. Create tmp japanese context files.
            ja_c_tmp = open(self.japanese_context_file_path, "w")
            with open(self.source_context_file_path,  newline='') as f:
                dataReader = csv.reader(f)
                header = next(dataReader)
                for row in dataReader:
                    if self.lang == "Ja":
                        if self.use_bing_translate == True or self.use_bing_translate == True:
                            context_sent = row[3]
                        else:
                            context_sent = \
                                tokenize_preprocess_japanese_sent(row[3])
                        ja_c_tmp.write(context_sent + "\n")
                    # For French and German, the tokenization and normalization
                    # would be executed later.
                    elif self.lang == "Fr":
                        ja_c_tmp.write(row[3].lower() + "\n")

            ja_c_tmp.close()
            if self.lang == "Fr":
                normalize_tokenized_sent(
                    self.japanese_context_file_path, self.lang)

            # 3. Get the translated results and attention scores.
            if self.use_google_translate:
                questions_sources = open(
                    self.japanese_question_file_path, "r").readlines()
                translated_questions = \
                    [google_translate(sentence, False, self.lang)
                     for sentence in questions_sources]
                translated_questions = \
                    [tokenize_preprocess_english_sent(
                        sentence) for sentence in translated_questions]
                translated_questions = [
                    q.replace("&#39;", "'").replace("\u200b\u200b", "") for q in translated_questions]

                google_trans_path = os.path.join("google_trans", self.lang)
                if not os.path.exists(google_trans_path):
                    os.makedirs(google_trans_path)
                self.question_trans_file_path = os.path.join(
                    google_trans_path, "TRANS.question.txt.new")

            elif self.use_bing_translate:
                questions_sources = open(
                    self.japanese_question_file_path, "r").readlines()
                translated_questions = \
                    [bing_translate(sentence, self.lang, 'en')
                     for sentence in questions_sources]

                translated_questions = \
                    [tokenize_preprocess_english_sent(
                        sentence) for sentence in translated_questions]
                translated_questions = [
                    q.replace("&#39;", "'").replace("\u200b\u200b", "") for q in translated_questions]

                bing_trans_path = os.path.join(
                    'trans_result', self.lang.lower(), 'v' + str(self.version), 'bing')
                if not os.path.exists(bing_trans_path):
                    os.makedirs(bing_trans_path)
                self.question_trans_file_path = os.path.join(
                    bing_trans_path, "TRANS.question.txt.new")

            else:
                if self.beam_search == True:
                    if self.soft == True:
                        translated_questions, attention_scores_questions = \
                            trans_from_files_beam(self.japanese_question_file_path, self.japanese_question_file_path,
                                                  self.trans_train_source, self.trans_train_target,
                                                  self.trans_embedding_model, self.trans_encdec_model, seed, 5, True)
                        trans_dir = os.path.split(
                            self.question_trans_file_path)[0]
                        trans_beam_dir = os.path.join(trans_dir, 'beam')
                        if not os.path.exists(trans_beam_dir):
                            os.makedirs(trans_beam_dir)
                        self.question_trans_file_path = os.path.join(
                            trans_beam_dir, "TRANS.question.txt.new")

                    else:
                        trans_from_files_beam(self.japanese_question_file_path, self.japanese_question_file_path,
                                              self.trans_train_source, self.trans_train_target,
                                              self.trans_embedding_model, self.trans_encdec_model, seed, 5, False)

                        translated_questions = open(
                            "trans.txt", 'r').read().splitlines()
                        # Save translated questions.
                        trans_dir = os.path.split(
                            self.question_trans_file_path)[0]
                        trans_beam_dir = os.path.join(trans_dir, 'beam')
                        if not os.path.exists(trans_beam_dir):
                            os.makedirs(trans_beam_dir)
                        self.question_trans_file_path = os.path.join(
                            trans_beam_dir, "TRANS.question.txt.new")

                else:
                    translated_questions, attention_scores_questions = \
                        trans_from_files(self.japanese_question_file_path, self.japanese_question_file_path,
                                         self.trans_train_source, self.trans_train_target,
                                         self.trans_embedding_model, self.trans_encdec_model,
                                         seed, trans_mode=True, save_attention_weights=True,
                                         replace_UNK=self.replace_UNK)

            trans_q_lines = translated_questions

            question_trans = open(self.question_trans_file_path, "w")

            for question in trans_q_lines:
                question_trans.write(question + "\n")
            question_trans.close()

            # Context Trans
            if self.use_google_translate:
                context_sources = open(
                    self.japanese_context_file_path, "r").readlines()
                translated_context = \
                    [google_translate(sentence, False, self.lang)
                     for sentence in context_sources]
                translated_context = \
                    [tokenize_preprocess_english_sent(
                        sentence) for sentence in translated_context]
                translated_context = [
                    c.replace("&#39;", "'").replace("\u200b\u200b", "") for c in translated_context]

                google_trans_path = os.path.join("google_trans", self.lang)
                if not os.path.exists(google_trans_path):
                    os.makedirs(google_trans_path)
                self.context_trans_file_path = os.path.join(
                    google_trans_path, "TRANS.txt.new")

                trans_c_lines = translated_context

            if self.use_bing_translate:
                context_sources = open(
                    self.japanese_context_file_path, "r").readlines()

                bing_translate_result = [bing_translate(sentence, self.lang, 'en', True)
                                         for sentence in context_sources]
                translated_context = [result[0]
                                      for result in bing_translate_result]
                alignment_info = [result[1]
                                  for result in bing_translate_result]

                translated_context = [sentence.lower().replace(
                    "\n", "") for sentence in translated_context]

                bing_trans_path = os.path.join(
                    'trans_result', self.lang.lower(), 'v' + str(self.version), 'bing')
                if not os.path.exists(bing_trans_path):
                    os.makedirs(bing_trans_path)
                self.context_trans_file_path = os.path.join(
                    bing_trans_path, "TRANS.txt.new")
                self.context_attention_file_path = os.path.join(
                    bing_trans_path, "ATTN.txt.new")

                trans_c_lines = translated_context
                trans_a_lines = alignment_info

            else:
                if self.beam_search == True:
                    if self.soft == True:
                        translated_context, attention_scores_context = \
                            trans_from_files_beam(self.japanese_context_file_path, self.japanese_context_file_path,
                                                  self.trans_train_source, self.trans_train_target,
                                                  self.trans_embedding_model, self.trans_encdec_model, seed, 5, True)

                        trans_c_lines = translated_context
                        trans_a_lines = attention_scores_context

                        # Reset the saved dir name.
                        trans_dir = os.path.split(
                            self.question_trans_file_path)[0]
                        self.context_trans_file_path = os.path.join(
                            trans_dir, "TRANS.txt.new")
                        self.context_attention_file_path = os.path.join(
                            trans_dir, "ATTN.txt.new")
                    else:
                        trans_from_files_beam(self.japanese_context_file_path, self.japanese_context_file_path,
                                              self.trans_train_source, self.trans_train_target,
                                              self.trans_embedding_model, self.trans_encdec_model, seed, 5, False)

                        translated_context = open(
                            "trans.txt", 'r').read().splitlines()
                        attention_scores_context = open(
                            "attn.txt", 'r').read().splitlines()

                        # Reset the saved dir name.
                        trans_dir = os.path.split(
                            self.question_trans_file_path)[0]
                        self.context_trans_file_path = os.path.join(
                            trans_dir, "TRANS.txt.new")
                        self.context_attention_file_path = os.path.join(
                            trans_dir, "ATTN.txt.new")

                else:
                    translated_context, attention_scores_context = \
                        trans_from_files(self.japanese_context_file_path, self.japanese_context_file_path,
                                         self.trans_train_source, self.trans_train_target,
                                         self.trans_embedding_model, self.trans_encdec_model,
                                         seed, trans_mode=True, save_attention_weights=True,
                                         replace_UNK=self.replace_UNK)

                    trans_c_lines = translated_context
                    trans_a_lines = attention_scores_context

            if self.use_google_translate == False and self.use_bing_translate == False:
                trans_c_lines = translated_context
                trans_a_lines = attention_scores_context
                context_attention = open(self.context_attention_file_path, "w")
                context_trans = open(self.context_trans_file_path, "w")
                if self.beam_search == True and self.soft == False:
                    # save context
                    for trans_context in trans_c_lines:
                        context_trans.write(trans_context + "\n")
                    context_trans.close()

                    # save attention
                    for trans_attention_index in trans_a_lines:
                        context_attention.write(trans_attention_index + "\n")
                    context_trans.close()

                else:
                    for trans_context, attention_score in zip(trans_c_lines, attention_scores_context):
                        context_trans.write(trans_context + "\n")
                        for i in range(len(attention_score)):
                            attention_weight = [str(float(weight))
                                                for weight in attention_score[i]]
                            context_attention.write(
                                " ".join(attention_weight) + "\n")
                        context_attention.write("\n")
                    context_trans.close()
                    context_attention.close()
            else:
                context_trans = open(self.context_trans_file_path, "w")

                for trans_context in trans_c_lines:
                    context_trans.write(trans_context + "\n")

                if self.use_bing_translate == True:
                    context_attention = open(
                        self.context_attention_file_path, "w")

                    for trans_attention in trans_a_lines:
                        context_attention.write(trans_attention + "\n")

                    context_attention.close()

                context_trans.close()

        else:
            # This is for quick evaluation.
            # The translated context and questions are sved under the directory `trans_results`
            # and when the `--online_trans` option is set to False, the system loads the context
            # and questions which have been transalted beforehand.
            if self.use_google_translate == True:
                google_trans_path = os.path.join("google_trans", self.lang)
                if not os.path.exists(google_trans_path):
                    os.makedirs(google_trans_path)
                self.context_trans_file_path = os.path.join(
                    google_trans_path, "TRANS.txt.new")
                self.question_trans_file_path = os.path.join(
                    google_trans_path, "TRANS.question.txt.new")

            elif self.use_bing_translate == True:
                bing_trans_path = os.path.join(
                    'trans_result', self.lang.lower(), 'v' + str(self.version), 'bing')
                if not os.path.exists(bing_trans_path):
                    os.makedirs(bing_trans_path)
                self.context_trans_file_path = os.path.join(
                    bing_trans_path, "TRANS.txt.new")
                self.question_trans_file_path = os.path.join(
                    bing_trans_path, "TRANS.question.txt.new")

            if self.beam_search == True:
                trans_dir = os.path.split(self.question_trans_file_path)[0]
                trans_dir = os.path.join(trans_dir, "beam")
                self.question_trans_file_path = os.path.join(
                    trans_dir, "TRANS.question.txt.new")
                self.context_trans_file_path = os.path.join(
                    trans_dir, "TRANS.txt.new")

            trans_context_f = open(self.context_trans_file_path)
            trans_context = trans_context_f.read()
            trans_context_f.close()
            trans_c_lines = trans_context.split('\n')

            trans_question_f = open(self.question_trans_file_path)
            trans_question = trans_question_f.read()
            trans_question_f.close()
            trans_q_lines = trans_question.split('\n')

        with open(self.question_file_path,  newline='') as f:
            dataReader = csv.reader(f)
            header = next(dataReader)
            for row in dataReader:
                question_id, title, paragraph_id, quastion = int(row[0]), row[1], int(
                    row[2]), row[3]
                if self.lang == "Ja":
                    answer_texts = [tokenize_preprocess_japanese_sent(row[4]), tokenize_preprocess_japanese_sent(
                        row[5]), tokenize_preprocess_japanese_sent(row[6])]
                elif self.lang == "Fr" or self.lang == "De":
                    answer_texts = normalize_tokenized_answers(
                        row[4], row[5], row[6], self.lang)

                sent_indices = sent_idx_dic[title][paragraph_id]
                paragraph_tokens = []

                for sent_idx in sent_indices:
                    paragraph_tokens.extend(trans_c_lines[sent_idx].split())

                paragraph = " ".join(paragraph_tokens)

                tokenized_paragraph = self._tokenizer.tokenize(paragraph)
                question_text = trans_q_lines[question_id]
                instance = self.text_to_instance(question_text,
                                                 paragraph,
                                                 answer_texts,
                                                 tokenized_paragraph,
                                                 question_id)
                yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text,
                         passage_text,
                         answer_texts=None,
                         passage_tokens=None,
                         question_id=None):
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans = []

        passage_offsets = [(token.idx, token.idx + len(token.text))
                           for token in passage_tokens]

        return make_reading_comprehension_instance(self._tokenizer.tokenize(question_text),
                                                   passage_tokens,
                                                   self._token_indexers,
                                                   passage_text,
                                                   token_spans,
                                                   answer_texts,
                                                   question_id)

    @classmethod
    def from_params(cls, params):
        dataset_type = params.pop("type")
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, token_indexers=token_indexers, lazy=lazy)


def make_reading_comprehension_instance(question_tokens,
                                        passage_tokens,
                                        token_indexers,
                                        passage_text,
                                        token_spans=None,
                                        answer_texts=None,
                                        question_id=None,
                                        additional_metadata=None):
    additional_metadata = additional_metadata or {}
    fields = {}
    passage_offsets = [(token.idx, token.idx + len(token.text))
                       for token in passage_tokens]

    # This is separate so we can reference it later with a known type.
    passage_field = TextField(passage_tokens, token_indexers)
    fields['passage'] = passage_field
    fields['question'] = TextField(question_tokens, token_indexers)
    metadata = {
        'original_passage': passage_text,
        'token_offsets': passage_offsets,
        'question_tokens': [token.text for token in question_tokens],
        'passage_tokens': [token.text for token in passage_tokens],
    }

    if answer_texts:
        metadata['answer_texts'] = answer_texts
    if question_id:
        metadata['question_id'] = str(question_id)
    else:
        metadata['question_id'] = '0'

    if token_spans:
        # There may be multiple answer annotations, so we pick the one that occurs the most.  This
        # only matters on the SQuAD dev set, and it means our computed metrics ("start_acc",
        # "end_acc", and "span_acc") aren't quite the same as the official metrics, which look at
        # all of the annotations.  This is why we have a separate official SQuAD metric calculation
        # (the "em" and "f1" metrics use the official script).
        candidate_answers: Counter = Counter()
        for span_start, span_end in token_spans:
            candidate_answers[(span_start, span_end)] += 1
        span_start, span_end = candidate_answers.most_common(1)[0][0]

        fields['span_start'] = IndexField(span_start, passage_field)
        fields['span_end'] = IndexField(span_end, passage_field)

    metadata.update(additional_metadata)
    fields['metadata'] = MetadataField(metadata)

    return Instance(fields)