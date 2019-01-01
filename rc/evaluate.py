from typing import Dict, Any, Iterable
import argparse
import logging
import os
import json

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.data import Instance, Vocabulary
from allennlp.common.params import Params
from model import BiDAFSelfAttention
import torch
from squad import SquadReaderEval

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Evaluate(Subcommand):
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

        subparser.add_argument('--unziped_archive_directory',
                               type=str,
                               default="default",
                               help='path to an unziped archived dorectory')

        subparser.add_argument('--elmo', action='store_true',
                               help='set True if you use ELMo for evaluation.')

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate(model,
             instances,
             data_iterator,
             cuda_device):
    model.eval()
    answer_json = {}

    iterator = data_iterator(instances, num_epochs=1,
                             cuda_device=cuda_device)
    logger.info("Iterating over dataset")
    generator_tqdm = Tqdm.tqdm(
        iterator, total=data_iterator.get_num_batches(instances))
    for batch in generator_tqdm:
        model(**batch)
        question_id = batch['metadata'][0]['question_id']
        metrics = model.get_metrics(True)
        answer_json[question_id] = metrics
        description = ', '.join(["%s: %.5f" % (name, value)
                                 for name, value in metrics.items()]) + " ||"
        generator_tqdm.set_description(description, refresh=False)

    with open('english_predict_metrics.json', 'w') as outfile:
        json.dump(answer_json, outfile)

    return model.get_metrics(reset=True)


def evaluate_from_args(args):
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger(
        'allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    # do not path any value for default one.
    if args.unziped_archive_directory != "default":
        if args.elmo == True:
            model, config = _load_elmo(args.unziped_archive_directory,
                                       args.archive_file, weights_file=None, cuda_device=args.cuda_device)
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

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop(
        'validation_dataset_reader', None)

    dataset_reader = SquadReaderEval.from_params(config.pop('dataset_reader'))

    evaluation_data_path = args.evaluation_data_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    iterator = DataIterator.from_params(config.pop("iterator"))
    iterator.index_with(model.vocab)

    metrics = evaluate(model, instances, iterator, args.cuda_device)

    logger.info("Finished evaluating.")
    logger.info("Metrics:")
    for key, metric in metrics.items():
        logger.info("%s: %s", key, metric)

    return metrics


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
