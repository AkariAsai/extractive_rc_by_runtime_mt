import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import nll_loss

from allennlp.common import Params, squad_eval
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from tri_linear_attention import TriLinearAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1, Average

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("bidaf_my")
class BidirectionalAttentionFlow(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model`
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).
    This implementation is based on allennlp's implementation with some minor modifications. 
    <https://github.com/allenai/allennlp/blob/master/allennlp/models/reading_comprehension/bidaf.py>`_

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 attention_similarity_function, modeling_layer, span_end_encoder,
                 dropout=0.2, mask_lstms=True, initializer=InitializerApplicator(),
                 regularizer=None):
        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)
        # Initialize layers.
        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))
        self._phrase_layer = phrase_layer
        self._matrix_attention = MatrixAttention(attention_similarity_function)
        self._modeling_layer = modeling_layer
        self._span_end_encoder = span_end_encoder

        # Inintialize start/end span predictors.
        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        span_start_input_dim = encoding_dim * 4 + modeling_dim
        self._span_start_predictor = \
            TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        span_end_encoding_dim = span_end_encoder.get_output_dim()
        span_end_input_dim = encoding_dim * 4 + span_end_encoding_dim
        self._span_end_predictor = \
            TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

        # Check dimentions
        check_dimensions_match(modeling_layer.get_input_dim(), 4 * encoding_dim,
                               "modeling layer input dim", "4 * encoding dim")
        check_dimensions_match(text_field_embedder.get_output_dim(), phrase_layer.get_input_dim(),
                               "text field embedder output dim", "phrase layer input dim")
        check_dimensions_match(span_end_encoder.get_input_dim(), 4 * encoding_dim + 3 * modeling_dim,
                               "span end encoder input dim", "4 * encoding dim + 3 * modeling dim")

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()

        # If dropout has been set, add Dropout layer.
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._mask_lstms = mask_lstms

        initializer(self)

    # Define and run the graph.
    def forward(self, question, passage, span_start=None, span_end=None, metadata=None):
        ######## 1/2. Embedding Layer ########
        # 2. After add embedding, pass the embedded vector to Highway network.
        embedded_question = self._highway_layer(
            self._text_field_embedder(question))
        embedded_passage = self._highway_layer(
            self._text_field_embedder(passage))

        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        ######## 3. Contextual Embedding Layer ########
        # Encode input vectors into new representation H and U by using a BiLSTM.
        # Shape: (batch_size, 2 * encoding dim, question_length)
        encoded_question = self._dropout(
            self._phrase_layer(embedded_question, question_lstm_mask))
        # Shape: (batch_size, 2 * encoding dim, paragraph_length)
        encoded_passage = self._dropout(
            self._phrase_layer(embedded_passage, passage_lstm_mask))
        # get each token dim by accessing the last dimention.
        encoding_dim = encoded_question.size(-1)

        ######## 4. Attention Flow Layer ########
        # Calculate similarity matrix for attention layer.
        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(
            encoded_passage, encoded_question)

        # Create weighted vectors attended by context to query attention.
        # Calculate C2Q(context to query) attention, 'a'
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(
            passage_question_similarity, question_mask)
        # Weighted vector by C2Q attentions. \hat{U}_:t \sum_j a_{tj} U_{:j}
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(
            encoded_question, passage_question_attention)

        # Create weighted vectors attended by query to context attention.
        # Replaced masked values to avoid let them affect the result.
        masked_similarity = util.replace_masked_values(
            passage_question_similarity,
            question_mask.unsqueeze(1), -1e7)

        # Calculate Q2C(query to context) attention, 'b'
        # Shape: (batch_size, passage_length)
        question_passage_similarity = \
            masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Pass to the softmax layer.
        # Shape: (batch_size, passage_length)
        question_passage_attention = \
            util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(
            encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Merge attention vectors
        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        ######## 5. Modeling Layer ########
        # Model query-aware context vector by BiLSTM.
        modeled_passage = self._dropout(self._modeling_layer(
            final_merged_passage, passage_lstm_mask))
        modeling_dim = modeled_passage.size(-1)

        ######## 6. Output Layer ########
        # Obtain the probability distribution of the start index.
        # Concat G(from attention flow layer) and M(from modeling layer)
        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        span_start_input = self._dropout(
            torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # Calculate the logits.
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(
            span_start_input).squeeze(-1)
        # Calculate Softmax (eq. 3)
        # Shape: (batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        # Obtain the probability distribution of the end index.
        # Shape: (batch_size, modeling_dim)
        span_start_representation = util.weighted_sum(
            modeled_passage, span_start_probs)
        # Shape: (batch_size, passage_length, modeling_dim)
        tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
                                                                                   passage_length,
                                                                                   modeling_dim)
        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim *
        # 3)
        span_end_representation = torch.cat([final_merged_passage,
                                             modeled_passage,
                                             tiled_start_representation,
                                             modeled_passage * tiled_start_representation],
                                            dim=-1)
        # Obtain new modling representation based on start probalility.
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_span_end = self._dropout(self._span_end_encoder(
            span_end_representation, passage_lstm_mask))
        # Concat attention flow output G and new modeling representation M2
        # Shape: (batch_size, passage_length, encoding_dim * 4 +
        # span_end_encoding_dim)
        span_end_input = self._dropout(
            torch.cat([final_merged_passage, encoded_span_end], dim=-1))
        # Shape: (batch_size, passage_length)
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
        # Shape: (batch_size, passage_length)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

        # Replace the masked values pretty small not to influence the final
        # results.
        span_start_logits = util.replace_masked_values(
            span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(
            span_end_logits, passage_mask, -1e7)

        best_span = self.get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "passage_question_attention": passage_question_attention,
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_span,
        }

        if span_start is not None:
            # Calculate the loss.
            # The training loss is the sum of the negative log probablities of
            # the true start and end indices by the predicted distributions,
            # everaged over all examples.
            loss = nll_loss(util.masked_log_softmax(
                span_start_logits, passage_mask), span_start.squeeze(-1))
            self._span_start_accuracy(
                span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(util.masked_log_softmax(span_end_logits,
                                                     passage_mask), span_end.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            # Why need to be `torch.stack`
            self._span_accuracy(best_span, torch.stack(
                [span_start, span_end], -1))
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].data.cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
        return output_dict

    def get_metrics(self, reset=False):
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }

    @staticmethod
    def get_best_span(span_start_logits, span_end_logits):
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError(
                "Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        # The best word span is the Variable of the shape (batchsize,2)
        best_word_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size, 2).fill_(0)).long()

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span

    @classmethod
    # Return a instance of BidirectionalAttentionFlow model.
    def from_params(cls, vocab, params):
        model = params.pop("type")
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params)
        num_highway_layers = params.pop_int("num_highway_layers")
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        similarity_function = SimilarityFunction.from_params(
            params.pop("similarity_function"))
        modeling_layer = Seq2SeqEncoder.from_params(
            params.pop("modeling_layer"))
        span_end_encoder = Seq2SeqEncoder.from_params(
            params.pop("span_end_encoder"))
        dropout = params.pop_float('dropout', 0.2)

        initializer = InitializerApplicator.from_params(
            params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(
            params.pop('regularizer', []))

        mask_lstms = params.pop_bool('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   num_highway_layers=num_highway_layers,
                   phrase_layer=phrase_layer,
                   attention_similarity_function=similarity_function,
                   modeling_layer=modeling_layer,
                   span_end_encoder=span_end_encoder,
                   dropout=dropout,
                   mask_lstms=mask_lstms,
                   initializer=initializer,
                   regularizer=regularizer)


@Model.register("bidaf_self")
class BiDAFSelfAttention(Model):
    """
    This class implements Christopher Clark's `Simple and Effective Multi-Paragraph Reading Comprehension
    <https://arxiv.org/abs/1710.10723>`_ (BiDAF + Self Attention)
    for multi paragraph reading comprehensions.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    residual_encoder : ``SimilarityFunction``
        The encoder that we will use after merging the representations from attention layers.
    span_start_encoder : ``Seq2SeqEncoder``
        The encoder that we will use for predicting span start.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    """
    def __init__(self, vocab,
                 text_field_embedder,
                 phrase_layer,
                 residual_encoder,
                 span_start_encoder,
                 span_end_encoder,
                 initializer,
                 dropout=0.2,
                 mask_lstms=True):
        super(BiDAFSelfAttention, self).__init__(vocab)
        # Initialize layers.
        self._text_field_embedder = text_field_embedder

        self._phrase_layer = phrase_layer
        # Inintialize start/end span predictors.
        encoding_dim = phrase_layer.get_output_dim()

        self._matrix_attention = TriLinearAttention(encoding_dim)
        self._merge_atten = TimeDistributed(
            torch.nn.Linear(encoding_dim * 4, encoding_dim))

        self._residual_encoder = residual_encoder
        self._self_atten = TriLinearAttention(encoding_dim)
        self._merge_self_atten = TimeDistributed(
            torch.nn.Linear(encoding_dim * 3, encoding_dim))

        self._span_start_encoder = span_start_encoder
        self._span_end_encoder = span_end_encoder

        self._span_start_predictor = TimeDistributed(
            torch.nn.Linear(encoding_dim, 1))
        self._span_end_predictor = TimeDistributed(
            torch.nn.Linear(encoding_dim, 1))

        initializer(self)

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        self._official_em = Average()
        self._official_f1 = Average()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
            # self._dropout = VariationalDropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms
    # Define and run the graph.

    def forward(self,  # type: ignore
                question,
                passage,
                span_start=None, span_end=None,
                metadata=None):
        ######## 1/2. Embedding Layer ########
        # 2. After add embedding, pass the embedded vector to Highway network.
        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(self._text_field_embedder(passage))

        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        ######## 3. Contextual Embedding Layer ########
        # Encode input vectors into new representation H and U by using a BiLSTM.
        # Shape: (batch_size, 2 * encoding dim, question_length)
        encoded_question = self._dropout(
            self._phrase_layer(embedded_question, question_lstm_mask))
        # Shape: (batch_size, 2 * encoding dim, paragraph_length)
        encoded_passage = self._dropout(
            self._phrase_layer(embedded_passage, passage_lstm_mask))
        # get each token dim by accessing the last dimention.
        encoding_dim = encoded_question.size(-1)

        ######## 4. Attention Flow Layer ########
        # Calculate similarity matrix for attention layer.
        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(
            encoded_passage, encoded_question)

        # Create weighted vectors attended by context to query attention.
        # Calculate C2Q(context to query) attention, 'a'
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(
            passage_question_similarity, question_mask)
        # Weighted vector by C2Q attentions. \hat{U}_:t \sum_j a_{tj} U_{:j}
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(
            encoded_question, passage_question_attention)

        # Create weighted vectors attended by query to context attention.
        # Replaced masked values to avoid let them affect the result.
        masked_similarity = util.replace_masked_values(
            passage_question_similarity,
            question_mask.unsqueeze(1), -1e7)

        # Calculate Q2C(query to context) attention, 'b'
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0]
        # Pass to the softmax layer.
        # Shape: (batch_size, passage_length)
        question_passage_attention = \
            util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(
            encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Merge attention vectors
        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        # Add purple "linear ReLU layer"
        final_merged_passage = F.relu(self._merge_atten(final_merged_passage))
        # Bi-GRU in the paper
        residual_layer = self._dropout(self._residual_encoder(
            self._dropout(final_merged_passage), passage_mask))
        self_atten_matrix = self._self_atten(residual_layer, residual_layer)

        # Expand mask for self-attention
        mask = (passage_mask.resize(batch_size, passage_length, 1) *
                passage_mask.resize(batch_size, 1, passage_length))

        # Mask should have zeros on the diagonal.
        # torch.eye does not have a gpu implementation, so we are forced to use
        # the cpu one and .cuda(). Not sure if this matters for performance.
        eye = torch.eye(passage_length, passage_length)
        if mask.is_cuda:
            eye = eye.cuda()
        self_mask = Variable(eye).resize(1, passage_length, passage_length)
        mask = mask * (1 - self_mask)

        self_atten_probs = util.last_dim_softmax(self_atten_matrix, mask)

        # Batch matrix multiplication:
        # (batch, passage_len, passage_len) * (batch, passage_len, dim) -> (batch, passage_len, dim)
        self_atten_vecs = torch.matmul(self_atten_probs, residual_layer)

        # (extended_batch_size, passage_length, embedding_dim * 3)
        concatenated = torch.cat([self_atten_vecs, residual_layer, residual_layer * self_atten_vecs],
                                 dim=-1)
        # _merge_self_atten => (extended_batch_size, passage_length,
        # embedding_dim)
        residual_layer = F.relu(self._merge_self_atten(concatenated))

        # print("residual", residual_layer.size())

        final_merged_passage += residual_layer
        final_merged_passage = self._dropout(final_merged_passage)

        # Bi-GRU in paper
        start_rep = self._span_start_encoder(
            final_merged_passage, passage_lstm_mask)
        span_start_logits = self._span_start_predictor(start_rep).squeeze(-1)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        end_rep = self._span_end_encoder(
            torch.cat([final_merged_passage, start_rep], dim=-1), passage_lstm_mask)
        span_end_logits = self._span_end_predictor(end_rep).squeeze(-1)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

        span_start_logits = util.replace_masked_values(
            span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(
            span_end_logits, passage_mask, -1e7)

        best_span = self.get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "passage_question_attention": passage_question_attention,
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_span,
        }

        if span_start is not None:
            # Calculate the loss.
            # The training loss is the sum of the negative log probablities of
            # the true start and end indices by the predicted distributions,
            # everaged over all examples.
            loss = nll_loss(util.masked_log_softmax(
                span_start_logits, passage_mask), span_start.squeeze(-1))
            self._span_start_accuracy(
                span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(util.masked_log_softmax(span_end_logits,
                                                     passage_mask), span_end.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            # Why need to be `torch.stack`
            self._span_accuracy(best_span, torch.stack(
                [span_start, span_end], -1))
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].data.cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
        return output_dict

    def get_metrics(self, reset=False):
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            'em': exact_match,
            'f1': f1_score,
        }

    @staticmethod
    def get_best_span(span_start_logits, span_end_logits):
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError(
                "Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        # The best word span is the Variable of the shape (batchsize,2)
        best_word_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size, 2).fill_(0)).long()

        span_start_logits = span_start_logits.data.cpu().numpy()
        span_end_logits = span_end_logits.data.cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span

    @classmethod
    def from_params(cls, vocab, params):
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, embedder_params)
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        residual_encoder = Seq2SeqEncoder.from_params(
            params.pop("residual_encoder"))
        span_start_encoder = Seq2SeqEncoder.from_params(
            params.pop("span_start_encoder"))
        span_end_encoder = Seq2SeqEncoder.from_params(
            params.pop("span_end_encoder"))
        initializer = InitializerApplicator.from_params(
            params.pop("initializer", []))
        dropout = params.pop('dropout', 0.2)

        evaluation_json_file = params.pop('evaluation_json_file', None)
        if evaluation_json_file is not None:
            logger.warning(
                "the 'evaluation_json_file' model parameter is deprecated, please remove")

        mask_lstms = params.pop('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   phrase_layer=phrase_layer,
                   residual_encoder=residual_encoder,
                   span_start_encoder=span_start_encoder,
                   span_end_encoder=span_end_encoder,
                   initializer=initializer,
                   dropout=dropout,
                   mask_lstms=mask_lstms)
