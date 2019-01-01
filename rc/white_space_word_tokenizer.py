from typing import List

from overrides import overrides

from allennlp.common import Params
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_filter import WordFilter, PassThroughWordFilter
from allennlp.data.tokenizers.word_splitter import WordSplitter
from allennlp.data.tokenizers.word_stemmer import WordStemmer, PassThroughWordStemmer


@WordSplitter.register('just_spaces_idx')
class JustSpacesWordSplitterIdx(WordSplitter):
    """
    A ``WordSplitter`` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces. The original JustSpacesWordSplitter does not
    store the tokens' start indices, which are neccessary for SQuAD questions, and thus
    I implemented this customized class. 
    """
    @overrides
    def split_words(self, sentence):
        tokens = []
        begin = 0
        for word in sentence.split():
            tokens.append(Token(word, begin))
            begin += (len(word) + 1)

        return tokens

    @classmethod
    def from_params(cls, params):
        params.assert_empty(cls.__name__)
        return cls()


@Tokenizer.register("white_space")
class WhiteSpaceWordTokenizer(Tokenizer):
    """
    A ``WordTokenizer`` handles the splitting of strings into words as well as any desired
    post-processing (e.g., stemming, filtering, etc.).  Note that we leave one particular piece of
    post-processing for later: the decision of whether or not to lowercase the token.  This is for
    two reasons: (1) if you want to make two different casing decisions for whatever reason, you
    won't have to run the tokenizer twice, and more importantly (2) if you want to lowercase words
    for your word embedding, but retain capitalization in a character-level representation, we need
    to retain the capitalization here.
    Parameters
    ----------
    word_splitter : ``WordSplitter``, optional
        The :class:`WordSplitter` to use for splitting text strings into word tokens.  The default
        is to use the ``SpacyWordSplitter`` with default parameters.
    word_filter : ``WordFilter``, optional
        The :class:`WordFilter` to use for, e.g., removing stopwords.  Default is to do no
        filtering.
    word_stemmer : ``WordStemmer``, optional
        The :class:`WordStemmer` to use.  Default is no stemming.
    start_tokens : ``List[str]``, optional
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``List[str]``, optional
        If given, these tokens will be added to the end of every string we tokenize.
    language : ``str``, optional
        We use spacy to tokenize strings; this option specifies which language to use.  By default
        we use English.
    pos_tags : ``bool``, optional
        By default we do not load spacy's tagging model, to save loading time and memory.  Set this
        to ``True`` if you want to have access to spacy's POS tags in the returned tokens.
    parse : ``bool``, optional
        By default we do not load spacy's parsing model, to save loading time and memory.  Set this
        to ``True`` if you want to have access to spacy's dependency parse tags in the returned
        tokens.
    ner : ``bool``, optional
        By default we do not load spacy's parsing model, to save loading time and memory.  Set this
        to ``True`` if you want to have access to spacy's NER tags in the returned tokens.
    """

    def __init__(self,
                 word_splitter=None,
                 word_filter=PassThroughWordFilter(),
                 word_stemmer=PassThroughWordStemmer(),
                 start_tokens=None,
                 end_tokens=None):
        self._word_splitter = JustSpacesWordSplitterIdx()
        self._word_filter = word_filter
        self._word_stemmer = word_stemmer
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    @overrides
    def tokenize(self, text):
        words = self._word_splitter.split_words(text)
        return self._filter_and_stem(words)

    @overrides
    def batch_tokenize(self, texts):
        batched_words = self._word_splitter.batch_split_words(texts)
        return [self._filter_and_stem(words) for words in batched_words]

    def _filter_and_stem(self, words):
        # filtered_words = self._word_filter.filter_words(words)
        # Not to filter stop words to avoid the mis-alignment.
        filtered_words = words
        stemmed_words = [self._word_stemmer.stem_word(
            word) for word in filtered_words]
        for start_token in self._start_tokens:
            stemmed_words.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            stemmed_words.append(Token(end_token, -1))
        return stemmed_words

    @classmethod
    def from_params(cls, params):
        word_splitter = WordSplitter.from_params(
            params.pop('word_splitter', {}))
        word_filter = WordFilter.from_params(params.pop('word_filter', {}))
        word_stemmer = WordStemmer.from_params(params.pop('word_stemmer', {}))
        start_tokens = params.pop('start_tokens', None)
        end_tokens = params.pop('end_tokens', None)
        params.assert_empty(cls.__name__)
        return cls(word_splitter=word_splitter,
                   word_filter=word_filter,
                   word_stemmer=word_stemmer,
                   start_tokens=start_tokens,
                   end_tokens=end_tokens)
