from typing import Dict, Any, Iterable
import argparse
import logging
import json
import os
import ntpath
from collections import Counter

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from data_ml import SQuADReaderML
from squad_ml_answer_retreival import SQuADMLAnswerRetreival, SQuADMLAnswerRetreivalBing
from utils import extract_word, google_translate, google_translate_to_fr, tokenize_preprocess_japanese_sent, bing_translate
from allennlp.data import Instance, Vocabulary
from allennlp.common.params import Params
from model import BiDAFSelfAttention
import torch
from tqdm import tqdm

import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append( str(current_dir) + '/../' )
from nmt.test import trans_from_files_beam

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Evaluate_MLQA(Subcommand):
    def add_subparser(self, name, parser):
        # pylint: disable=protected-access
        description = '''Evaluate the specified model + dataset'''
        subparser = parser.add_parser(
            name, description=description, help='Evaluate the specified model + dataset')

        subparser.add_argument('archive_file', type=str,
                               help='path to an archived trained model')

        evaluation_data_file = subparser.add_mutually_exclusive_group(
            required=True)
        evaluation_data_file.add_argument('--evaluation-data-file',
                                          type=str,
                                          help='path to the file containing the evaluation data')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=0,
                                 help='id of GPU to use (if any)')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        # Add argument for different trans models.
        subparser.add_argument('--trans_embedding_model', type=str,
                               help='path to an archived NMT trained embedding model',
                               default='nmt/params/embedding.bin')
        subparser.add_argument('--trans_encdec_model', type=str,
                               help='path to an archived NMT trained encdec trained model',
                               default='nmt/params/encdec.bin')
        subparser.add_argument('--trans_train_source', type=str,
                               help='path to an archived NMT source embedding model',
                               default='nmt/corpus/enja_corpus/wiki_alignment_ja')
        subparser.add_argument('--trans_train_target', type=str,
                               help='path to an archived NMT train target file',
                               default='nmt/corpus/enja_corpus/wiki_alignment_en')
        
        # Add argument for different RC models. 
        subparser.add_argument('--unziped_archive_directory', type=str, default="default",
                               help='path to an unziped archived dorectory')

        subparser.add_argument('--elmo', action='store_true',
                               help='set True if you use ELMo for evaluation.')

        subparser.add_argument('--beam', action='store_true',
                               help='set True if you use beamsearch for translation.')
        subparser.add_argument(
            '-t', '--use_question_tag', action='store_false')
        subparser.add_argument(
            '-g', '--use_google_translate', action='store_true')
        subparser.add_argument(
            '-b', '--use_bing_translate', action='store_true')
        subparser.add_argument(
            '-r', '--replace_UNK', action='store_true')
        subparser.add_argument(
            '-l', '--language', default='Ja')
        subparser.add_argument(
            '-v', '--version', type=int, default=3)
        subparser.add_argument(
            '--online_trans', action='store_true')
        subparser.add_argument(
            '--soft', action='store_true')
        
        # added for back translate by our NMT.
        subparser.add_argument(
            '--back_trans_ours', action='store_true')
        subparser.add_argument(
            '--back_trans_bing', action='store_true')
        subparser.add_argument('--enja_emb', type=str,
                               help='path to an archived NMT embedding')
        subparser.add_argument('--enja_encdec', type=str,
                               help='path to an archived NMT encdec')
        subparser.add_argument('--enja_train_source', type=str,
                               help='path to an archived NMT train source file')
        subparser.add_argument('--enja_train_target', type=str,
                               help='path to an archived NMT train target file')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser

def evaluate_mlqa(model,
                  instances,
                  data_iterator,
                  cuda_device, lang, version=4, embedding_filepath='embedding',
                  beam_search=False, soft_alignment=False):
    embedding_name = ntpath.basename(embedding_filepath).split(".bin")[0]
    answer_retreival = SQuADMLAnswerRetreival(
        lang, version, embedding_name, beam_search, soft_alignment)
    model.eval()

    iterator = data_iterator(instances, num_epochs=1,
                             cuda_device=cuda_device)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(
        iterator, total=data_iterator.get_num_batches(instances))

    question_idx = []
    ground_truth_answers = {}
    predicted_ans = []
    predicted_ans_str = []
    ja_anss = {}
    en_anss = {}

    for batch in generator_tqdm:
        result = model(**batch)
        anss = result['best_span']
        ans_strs = result['best_span_str']
        num = len(batch["metadata"])

        for i in range(num):
            question_idx.append(batch["metadata"][i]["question_id"])
            ground_truth_answers[int(question_idx[-1])] = \
                batch["metadata"][i]["answer_texts"]
        for ans in ans_strs:
            predicted_ans_str.append(ans)
        for ans in anss:
            predicted_ans.append(ans)

        id2answerindices_dict = \
            answer_retreival.get_id2answerindices_dict(
                predicted_ans, question_idx)
        id2answer_dict = \
            answer_retreival.get_id2answerindices_dict(
                predicted_ans_str, question_idx)

        if soft_alignment == True:
            ja_anss.update(answer_retreival.get_japanese_answers_with_attention_use_soft_alignment(
                id2answerindices_dict))
        else:
            if beam_search == True:
                ja_anss.update(
                    answer_retreival.get_japanese_answers_with_attention_beam_index(id2answerindices_dict))
            else:
                ja_anss.update(answer_retreival.get_japanese_answers_with_attention(
                    id2answerindices_dict))

        for k, v in ja_anss.items():
            print("{0}:<JA>{1}, <EN>{2}".format(
                k, ja_anss[k], id2answer_dict[k]))

    save_path = 'japanese_ans_predicted.json'
    f = open(save_path, "w")
    json.dump(ja_anss, f)
    f.close()

    save_path_answer_in_trans = 'predicted_ans_english.json'

    f = open(save_path_answer_in_trans, "w")
    json.dump(id2answer_dict, f)
    f.close()

    id2answer_save_path = 'japanese_ans_predicted_indice.json'
    f = open(id2answer_save_path, "w")
    id2answerindices_dict = {k: (int(v.data[0]), int(v.data[1])) for k,
                             v in id2answerindices_dict.items()}
    json.dump(id2answerindices_dict, f)
    f.close()

    eval_dict = evaluate(ground_truth_answers, ja_anss, lang)
    print(eval_dict)

    return {"F1": eval_dict['f1'], "EM": eval_dict['exact_match']}

def evaluate_mlqa_back_trans_ours(model,
                  instances,
                  data_iterator,
                  cuda_device, lang, version, enja_emb,
                  enja_encdec, enja_train_source,enja_train_target):

    if not (enja_emb and enja_encdec and enja_train_source and enja_train_target):
        print("necessary file path is empty. Google Translate will be used for answer translation.")
        return(evaluate_mlqa_google_translate(model,
                                instances,
                                data_iterator,
                                cuda_device, lang, version))
    else:
        model.eval()
        iterator = data_iterator(instances, num_epochs=1,
                                cuda_device=cuda_device)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(
            iterator, total=data_iterator.get_num_batches(instances))

        question_idx = []
        ground_truth_answers = {}
        predicted_ans = []
        predicted_ans_str = []
        ja_anss = {}
        en_anss = {}
        id2answer_dict = {}

        for batch in generator_tqdm:
            result = model(**batch)
            anss = result['best_span']
            ans_strs = result['best_span_str']
            num = len(batch["metadata"])
            for i in range(num):
                question_idx.append(batch["metadata"][i]["question_id"])
                ground_truth_answers[int(question_idx[-1])
                                    ] = batch["metadata"][i]["answer_texts"]
            for ans in ans_strs:
                predicted_ans_str.append(ans)

            id2answer_batch = {int(id_): span
                            for id_, span in
                            zip(question_idx, predicted_ans_str)}
            
            id2answer_dict.update(id2answer_batch)

        ans_tmp_file_path = "/home/asai/BiDAF_PyTorch/ans_tmp.txt"
        ans_tmp_f = open(ans_tmp_file_path,"w")

        for i in range(0, len(id2answer_dict)):
            ans_tmp_f.write(id2answer_dict[i]+"\n")
        
        ans_tmp_f.close()

        translated_answers, _ = \
            trans_from_files(ans_tmp_file_path, ans_tmp_file_path,
                            enja_train_source, enja_train_target, enja_emb, enja_encdec,
                            3, trans_mode=True, save_attention_weights=False, replace_UNK=False)
        ja_anss = {}
        for i in tqdm(range(0, len(translated_answers))):
            ja_anss[i] = postprocess_back_trans(translated_answers[i])

        save_path = 'japanese_ans_predicted_back_trans.json'
        f = open(save_path, "w")
        json.dump(ja_anss, f)
        f.close()

        save_path_answer_in_trans = 'predicted_ans_english.json'

        f = open(save_path_answer_in_trans, "w")
        json.dump(id2answer_dict, f)
        f.close()

        eval_dict = evaluate(ground_truth_answers, ja_anss, lang)
        print(eval_dict)

        return {"F1": eval_dict['f1'], "EM": eval_dict['exact_match']}

def evaluate_mlqa_google_translate(model,
                                   instances,
                                   data_iterator,
                                   cuda_device, lang="Ja", version=3):
    model.eval()
    iterator = data_iterator(instances, num_epochs=1,
                             cuda_device=cuda_device)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(
        iterator, total=data_iterator.get_num_batches(instances))

    question_idx = []
    ground_truth_answers = {}
    predicted_ans = []
    predicted_ans_str = []
    ja_anss = {}
    en_anss = {}
    
    id2answer_dict = {}

    for batch in generator_tqdm:
        result = model(**batch)
        anss = result['best_span']
        ans_strs = result['best_span_str']
        num = len(batch["metadata"])
        for i in range(num):
            question_idx.append(batch["metadata"][i]["question_id"])
            ground_truth_answers[int(question_idx[-1])
                                 ] = batch["metadata"][i]["answer_texts"]
        for ans in ans_strs:
            predicted_ans_str.append(ans)

        id2answer_batch = {int(id_): span
                          for id_, span in
                          zip(question_idx, predicted_ans_str)}
        
        id2answer_dict.update(id2answer_batch)

    if lang == "Ja":
        ja_anss = {k: google_translate(v, toJa=True)
                    for k, v in id2answer_dict.items()}
        ja_anss = {k: tokenize_preprocess_japanese_sent(
            v) for k, v in ja_anss.items()}
    elif lang == "Fr":
        ja_anss = {k: google_translate_to_fr(v, True)
                    for k, v in id2answer_dict.items()}

    save_path = 'japanese_ans_predicted.json'
    f = open(save_path, "w")
    json.dump(ja_anss, f)
    f.close()

    eval_dict = evaluate(ground_truth_answers, ja_anss)
    print(eval_dict)

    return {"F1": eval_dict['f1'], "EM": eval_dict['exact_match']}


def evaluate_mlqa_bing_translate(model,
                  instances,
                  data_iterator,
                  cuda_device, lang, version=4, embedding_name="bing", back_trans=True):
    answer_retreival = SQuADMLAnswerRetreivalBing(lang, version, embedding_name)

    model.eval()

    iterator = data_iterator(instances, num_epochs=1,
                             cuda_device=cuda_device)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(
        iterator, total=data_iterator.get_num_batches(instances))

    question_idx = []
    ground_truth_answers = {}
    predicted_ans = []
    predicted_ans_str = []
    ja_anss = {}
    en_anss = {}

    id2answer_dict = {}

    if back_trans==False:
        for batch in generator_tqdm:
            result = model(**batch)
            anss = result['best_span']
            ans_strs = result['best_span_str']
            num = len(batch["metadata"])
            for i in range(num):
                question_idx.append(batch["metadata"][i]["question_id"])
                ground_truth_answers[int(question_idx[-1])] = \
                    batch["metadata"][i]["answer_texts"]
            for ans in ans_strs:
                predicted_ans_str.append(ans)
            for ans in anss:
                predicted_ans.append(ans)

            id2answerindices_dict = \
                answer_retreival.get_id2answerindices_dict(
                    predicted_ans, question_idx)
            id2answer_dict = \
                answer_retreival.get_id2answerindices_dict(
                    predicted_ans_str, question_idx)

            ja_anss.update(answer_retreival.get_japanese_answers_with_attention(
                id2answerindices_dict, id2answer_dict))
    else:
        for batch in generator_tqdm:
            result = model(**batch)
            anss = result['best_span']
            ans_strs = result['best_span_str']
            num = len(batch["metadata"])
            for i in range(num):
                question_idx.append(batch["metadata"][i]["question_id"])
                ground_truth_answers[int(question_idx[-1])
                                    ] = batch["metadata"][i]["answer_texts"]
            for ans in ans_strs:
                predicted_ans_str.append(ans)

            id2answer_batch = {int(id_): span
                            for id_, span in
                            zip(question_idx, predicted_ans_str)}
            
            id2answer_dict.update(id2answer_batch)

        if lang == "Ja":
            ja_anss = {k: bing_translate(v, "en", "ja")
                        for k, v in id2answer_dict.items()}
            ja_anss = {k: tokenize_preprocess_japanese_sent(
                v) for k, v in ja_anss.items()}
            print(ja_ans)
        elif lang == "Fr":
            ja_anss = {k: bing_translate(v, "en", "fr")
                        for k, v in id2answer_dict.items()}

    ja_anss = {k: tokenize_preprocess_japanese_sent(
        v) for k, v in ja_anss.items()}
    for k, v in ja_anss.items():
        print("{0}:<JA>{1}, <EN>{2}".format(k, ja_anss[k], id2answer_dict[k]))

    save_path = 'japanese_ans_predicted.json'
    f = open(save_path, "w")
    json.dump(ja_anss, f)
    f.close()

    save_path_answer_in_trans = 'predicted_ans_english.json'

    f = open(save_path_answer_in_trans, "w")
    json.dump(id2answer_dict, f)
    f.close()

    eval_dict = evaluate(ground_truth_answers, ja_anss, lang)
    print(eval_dict)

    return {"F1": eval_dict['f1'], "EM": eval_dict['exact_match']}

def evaluate_from_args(args):
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger(
        'allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
    print(args.version)

    # Load from archive
    if args.unziped_archive_directory != "default":
        if args.elmo == True:
            model, config = _load_elmo(args.unziped_archive_directory,
                                       args.archive_file, weights_file=None,
                                       cuda_device=args.cuda_device)
        else:
            model, config = _load(args.unziped_archive_directory, weights_file=None,
                                  cuda_device=args.cuda_device)
    else:
        archive = load_archive(args.archive_file, args.cuda_device,
                               args.overrides, args.weights_file)
        config = archive.config
        prepare_environment(config)
        model = archive.model

    model.eval()

    # Load the evaluation data
    # Load evaluation dataset for multilingual evaluation.
    # The validation_dataset are called from the achieved model,
    # so you need to reload the multilingual file.
    validation_dataset_reader_params = config.pop(
        'validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(
            validation_dataset_reader_params)
    else:
        dataset_reader = SQuADReaderML.from_params(
            config.pop('dataset_reader'))

        dataset_reader.set_nmt_models_resources(args.trans_embedding_model, args.trans_encdec_model,
                                                args.trans_train_source, args.trans_train_target,
                                                args.use_question_tag, args.replace_UNK,
                                                args.version, args.online_trans, args.beam, args.soft)

        if args.language == "Fr":
            dataset_reader.set_squad_test_resources_fr()
        elif args.language == "Ja":
            dataset_reader.set_squad_test_resources_ja()

        dataset_reader.set_google_translate_mode(args.use_google_translate)
        dataset_reader.set_bing_translate_mode(args.use_bing_translate)
    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)

    # TODO: Fix this file path argument, it is not used and misleading.
    instances = dataset_reader.read("inputdata_tmp/ja_question_v2.csv")
    iterator = DataIterator.from_params(config.pop("iterator"))
    iterator.index_with(model.vocab)

    if args.use_google_translate:
        metrics = evaluate_mlqa_google_translate(
            model, instances, iterator, args.cuda_device, args.language, args.version)
    
    elif args.use_bing_translate:
        metrics = evaluate_mlqa_bing_translate(
            model, instances, iterator, args.cuda_device, args.language, args.version, "bing", args.back_trans_bing)

    else:
        if args.back_trans_ours == True:
            metrics = evaluate_mlqa_back_trans_ours(\
                model, instances, iterator, args.cuda_device, args.language,\
                args.version, args.enja_emb, args.enja_encdec,args.enja_train_source,args.enja_train_target)
        else:
            metrics = evaluate_mlqa(
                model, instances, iterator, args.cuda_device, args.language,
                args.version, args.trans_embedding_model, args.beam, args.soft)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics


# This is codes to evaluate usinng official squad scripts.
def exact_match_score(prediction, ground_truth, lang="Ja", lower=False):
    if lang == "Ja":
        return prediction.split() == tokenize_preprocess_japanese_sent(ground_truth).split()
    else:
        return prediction.split() == ground_truth.lower().split() or prediction == ground_truth.lower()


def f1_score(prediction, ground_truth, lang="Ja", lower=False):
    prediction_tokens = prediction.split()
    if lang == 'Ja':
        ground_truth_tokens = tokenize_preprocess_japanese_sent(
            ground_truth).split()
    else:
        ground_truth_tokens = ground_truth.lower().split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    if prediction == ground_truth.lower():
        return 1.0
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, lang="Ja"):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, lang=lang)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(ground_truth_answers, preds, lang="Ja"):
    eval_dic = {}
    f1 = exact_match = 0
    total = len(preds) - 1
    count = 0
    for i in range(len(preds)):
        ground_truth_answer = ground_truth_answers[i]
        pred = preds[i]
        if lang == "Fr" or lang == "De":
            pred = pred.replace("’", "'")
            pred = pred.replace('"', "'")
            ground_truth_answer = [answer.replace("’", "'").replace(
                '"', "'") for answer in ground_truth_answer]

        exact_match += metric_max_over_ground_truths(
            exact_match_score, pred, ground_truth_answer)
        f1 += metric_max_over_ground_truths(
            f1_score, pred, ground_truth_answer)

        eval_dic[i] = {'f1': metric_max_over_ground_truths(
            f1_score, pred, ground_truth_answer), 'pred': "".join(pred)}

    # exact_match = 100.0 * exact_match / total
    # f1 = 100.0 * f1 / total
    exact_match = 100.0 * exact_match / len(preds)
    f1 = 100.0 * f1 / len(preds)

    with open(lang + '_predict_metrics.json', 'w') as outfile:
        json.dump(eval_dic, outfile)

    return {'exact_match': exact_match, 'f1': f1}


# Define achieve again to avoid loading errors.
def _load(serialization_dir, weights_file=None, cuda_device=-1):
    config = Params.from_file(os.path.join(
        serialization_dir, 'config.json'), "")
    config.loading_from_archive = True
    weights_file = os.path.join(serialization_dir, 'weights.th')
    vocab_dir = os.path.join(serialization_dir, 'vocabulary')
    vocab = Vocabulary.from_files(vocab_dir)
    model_params = config.get('model')
    remove_pretrained_embedding_params(model_params)

    model = BiDAFSelfAttention.from_params(vocab=vocab, params=model_params)
    model_state = torch.load(
        weights_file, map_location=device_mapping(cuda_device))
    model.load_state_dict(model_state)

    if cuda_device >= 0:
        model.cuda(cuda_device)
    else:
        model.cpu()

    return model, config


def remove_pretrained_embedding_params(params):
    keys = params.keys()
    if 'pretrained_file' in keys:
        del params['pretrained_file']
    for value in params.values():
        if isinstance(value, Params):
            remove_pretrained_embedding_params(value)


def device_mapping(cuda_device):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """
    def inner_device_mapping(storage, location):  # pylint: disable=unused-argument
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage
    return inner_device_mapping


def _load_elmo(serialization_dir, vocabulary_dir, weights_file=None, cuda_device=-1):
    config = Params.from_file(os.path.join(
        serialization_dir, 'config.json'), "")
    config.loading_from_archive = True
    weights_file = os.path.join(vocabulary_dir, 'best.th')

    vocab_dir = os.path.join(vocabulary_dir, 'vocabulary')
    vocab = Vocabulary.from_files(vocab_dir)
    model_params = config.get('model')
    remove_pretrained_embedding_params(model_params)

    model = BiDAFSelfAttention.from_params(vocab=vocab, params=model_params)
    model_state = torch.load(
        weights_file, map_location=device_mapping(cuda_device))
    model.load_state_dict(model_state)

    if cuda_device >= 0:
        model.cuda(cuda_device)
    else:
        model.cpu()

    return model, config

def get_id2answerindices_dict(answer_spans, question_ids):
    id2answerindices_dict = {int(id_): span
                                for id_, span in
                                zip(question_ids, answer_spans)}
    return id2answerindices_dict

def postprocess_back_trans(ans):
    answer_tokens = []
    tokenized_ans = ans.split()
    for token in tokenized_ans:
        if token not in answer_tokens:
            answer_tokens.append(token)
    
    return " ".join(answer_tokens)