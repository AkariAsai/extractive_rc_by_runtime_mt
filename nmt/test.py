from nmt.data import Corpus
from nmt.model import Embedding
from nmt.model import EncDec
import nmt.utils as utils

from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import os
import time
import sys

minFreqSource = 5
minFreqTarget = 5
hiddenDim = 512
decay = 0.5
gradClip = 1.0
dropoutRate = 0.2
numLayers = 1

# use sentence pairs whose maximum lengths are 100 in both source and
# target sides
maxLen = 100
maxEpoch = 20
decayStart = 5

sourceEmbedDim = hiddenDim
targetEmbedDim = hiddenDim

learningRate = 1.0
momentumRate = 0.75

gpuId = [0]
seed = 3

device = torch.device('cuda:' + str(gpuId[0]))
cpu = torch.device('cpu')

weightDecay = 1.0e-06

batchSize = 1

torch.set_num_threads(1)

torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)


def trans_from_files_beam(trans_source, trans_target, train_source, train_target,
                          embedding_params, encdec_params, seed=3, beam_size=10, use_attenMatrix=False):
    batchSize = 1
    beamSize = beam_size
    print('trans_source :' + trans_source)
    print('train_source :' + train_source)
    print('train_target :' + train_target)
    print('embedding_params :' + embedding_params)
    print('encdec_params :' + encdec_params)

    corpus = Corpus(train_source, train_source, train_target, trans_source,
                    trans_source, trans_target, minFreqSource, minFreqTarget, maxLen)

    print('Source vocabulary size: ' + str(corpus.sourceVoc.size()))
    print('Target vocabulary size: ' + str(corpus.targetVoc.size()))
    print('# of training samples: ' + str(len(corpus.trainData)))
    print('# of develop samples:  ' + str(len(corpus.devData)))
    print('SEED: ', str(seed))

    embedding = Embedding(sourceEmbedDim, targetEmbedDim,
                          corpus.sourceVoc.size(), corpus.targetVoc.size())
    encdec = EncDec(sourceEmbedDim, targetEmbedDim, hiddenDim,
                    corpus.targetVoc.size(), dropoutRate=dropoutRate, numLayers=numLayers)

    encdec.wordPredictor.softmaxLayer.weight = embedding.targetEmbedding.weight
    encdec.wordPredictor = nn.DataParallel(encdec.wordPredictor, gpuId)

    batchListTrain = utils.buildBatchList(len(corpus.trainData), batchSize)
    batchListDev = utils.buildBatchList(len(corpus.devData), batchSize)

    withoutWeightDecay = []
    withWeightDecay = []
    for name, param in list(embedding.named_parameters()) + list(encdec.named_parameters()):
        if 'bias' in name or 'Embedding' in name:
            withoutWeightDecay += [param]
        elif 'softmax' not in name:
            withWeightDecay += [param]
    optParams = [{'params': withWeightDecay, 'weight_decay': weightDecay},
                 {'params': withoutWeightDecay, 'weight_decay': 0.0}]
    totalParamsNMT = withoutWeightDecay + withWeightDecay

    opt = optim.SGD(optParams, momentum=momentumRate, lr=learningRate)

    bestDevGleu = -1.0
    prevDevGleu = -1.0

    torch.set_grad_enabled(False)

    embedding.load_state_dict(torch.load(embedding_params))
    encdec.load_state_dict(torch.load(encdec_params))

    embedding.to(device)
    encdec.to(device)

    embedding.eval()
    encdec.eval()

    f_trans = open('./trans.txt', 'w')
    f_gold = open('./gold.txt', 'w')
    f_attn = open('./attn.txt', 'w')

    devPerp = 0.0
    totalTokenCount = 0.0

    attention_scores_matrix_list = []
    trans_result = []

    for batch in tqdm(batchListDev):
        batchSize = batch[1] - batch[0] + 1

        batchInputSource, lengthsSource, batchInputTarget, batchTarget, lengthsTarget, tokenCount, batchData = corpus.processBatchInfoNMT(
            batch, train=False, device=device)

        inputSource = embedding.getBatchedSourceEmbedding(batchInputSource)
        sourceH, (hn, cn) = encdec.encode(inputSource, lengthsSource)

        if beamSize == 1:
            indicesGreedy, lengthsGreedy, attentionIndices = encdec.greedyTrans(
                corpus.targetVoc.bosIndex, corpus.targetVoc.eosIndex, lengthsSource, embedding.targetEmbedding, sourceH, (hn, cn), device, maxGenLen=maxLen)
        else:
            if use_attenMatrix == True:
                indicesGreedy, lengthsGreedy, attentionIndices, attenMatrix = encdec.beamSearchAttn(
                    corpus.targetVoc.bosIndex, corpus.targetVoc.eosIndex, lengthsSource, embedding.targetEmbedding, sourceH, (hn, cn), device, beamSize=beamSize, maxGenLen=maxLen)
                attention_scores_matrix_list.append(attenMatrix)
            else:
                indicesGreedy, lengthsGreedy, attentionIndices = encdec.beamSearch(
                    corpus.targetVoc.bosIndex, corpus.targetVoc.eosIndex, lengthsSource, embedding.targetEmbedding, sourceH, (hn, cn), device, beamSize=beamSize, maxGenLen=maxLen)
        indicesGreedy = indicesGreedy.to(cpu)

        for i in range(batchSize):
            target_tokens = []
            for k in range(lengthsGreedy[i] - 1):
                index = indicesGreedy[i, k].item()
                if index == corpus.targetVoc.unkIndex:
                    index = attentionIndices[i, k].item()
                    target_tokens.append(
                        batchData[i].sourceOrigStr[index] + ' ')
                    f_trans.write(batchData[i].sourceOrigStr[index] + ' ')
                else:
                    f_trans.write(corpus.targetVoc.tokenList[index].str + ' ')
                    target_tokens.append(
                        corpus.targetVoc.tokenList[index].str + ' ')

            trans_result.append(" ".join(target_tokens))
            f_trans.write('\n')

            for k in range(lengthsTarget[i] - 1):
                index = batchInputTarget[i, k + 1].item()
                if index == corpus.targetVoc.unkIndex:
                    f_gold.write(batchData[i].targetUnkMap[k] + ' ')
                else:
                    f_gold.write(corpus.targetVoc.tokenList[index].str + ' ')
            f_gold.write('\n')

            # This is code to save attention max index.
            for k in range(lengthsGreedy[i] - 1):
                f_attn.write(str(attentionIndices[i, k].item()) + ' ')
            f_attn.write('\n')

        batchInputTarget = batchInputTarget.to(device)
        batchTarget = batchTarget.to(device)
        inputTarget = embedding.getBatchedTargetEmbedding(batchInputTarget)

        loss = encdec(inputTarget, lengthsTarget, lengthsSource,
                      (hn, cn), sourceH, batchTarget)
        loss = loss.sum()
        devPerp += loss.item()

        totalTokenCount += tokenCount

    f_trans.close()
    f_gold.close()
    f_attn.close()

    if use_attenMatrix == True:
        return trans_result, attention_scores_matrix_list

# This is an older version of trans_from_files implementation. 
# TODO: integrate codes with trans_from_files_beam
def trans_from_files(trans_source, trans_target, train_source, train_target,
                     embedding_params, encdec_params, seed,
                     trans_mode=False, save_attention_weights=False,
                     replace_UNK=False):
    print('trans_source :' + trans_source)
    print('train_source :' + train_source)
    print('train_target :' + train_target)
    print('embedding_params :' + embedding_params)
    print('encdec_params :' + encdec_params)

    if replace_UNK == True:
        lang_links_f = open(
            "/home/dl-exp/wiki_process/category/result/enja_langlinks.json", "r")
        langlinks_dict = json.load(lang_links_f)

    torch.set_num_threads(1)
    batchSize = 1

    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_device(gpuId[0])
    torch.cuda.manual_seed(seed)

    corpus = Corpus(train_source, train_source, train_target, trans_source,
                    trans_source, trans_target, minFreqSource, minFreqTarget, maxLen)

    print('Source vocabulary size: ' + str(corpus.sourceVoc.size()))
    print('Target vocabulary size: ' + str(corpus.targetVoc.size()))
    print()
    print('# of training samples: ' + str(len(corpus.trainData)))
    print('# of develop samples:  ' + str(len(corpus.devData)))
    print('SEED: ', str(seed))
    print()

    embedding = Embedding(sourceEmbedDim, targetEmbedDim,
                          corpus.sourceVoc.size(), corpus.targetVoc.size())
    encdec = EncDec(sourceEmbedDim, targetEmbedDim, hiddenDim, corpus.targetVoc.size(),
                    dropoutRate=dropoutRate, numLayers=numLayers)

    encdec.wordPredictor.softmaxLayer.weight = embedding.targetEmbedding.weight
    encdec.wordPredictor = nn.DataParallel(encdec.wordPredictor, gpuId)

    batchListTrain = utils.buildBatchList(len(corpus.trainData), batchSize)
    batchListDev = utils.buildBatchList(len(corpus.devData), batchSize)

    withoutWeightDecay = []
    withWeightDecay = []
    for name, param in list(embedding.named_parameters()) + list(encdec.named_parameters()):
        if 'bias' in name or 'Embedding' in name:
            withoutWeightDecay += [param]
        elif 'softmax' not in name:
            withWeightDecay += [param]
    optParams = [{'params': withWeightDecay, 'weight_decay': weightDecay},
                 {'params': withoutWeightDecay, 'weight_decay': 0.0}]
    totalParamsNMT = withoutWeightDecay + withWeightDecay

    opt = optim.SGD(optParams, momentum=momentumRate, lr=learningRate)

    bestDevGleu = -1.0
    prevDevGleu = -1.0

    # Load trained weights
    embedding.load_state_dict(torch.load(embedding_params))
    encdec.load_state_dict(torch.load(encdec_params))

    embedding.cuda()
    encdec.cuda()

    # Set evaluation mode
    embedding.eval()
    encdec.eval()

    trans_result = []
    attention_scores_matrix_list = []
    f_trans = open('./trans.txt', 'w')

    if trans_mode == False:
        f_gold = open('./gold.txt', 'w')

    devPerp = 0.0
    totalTokenCount = 0.0

    for batch in batchListDev:
        batchSize = batch[1] - batch[0] + 1
        batchInputSource, lengthsSource, batchInputTarget, batchTarget, \
            lengthsTarget, tokenCount, batchData = corpus.processBatchInfoNMT(
                batch, train=False, volatile=True)

        inputSource = embedding.getBatchedSourceEmbedding(batchInputSource)
        sourceH, (hn, cn) = encdec.encode(inputSource, lengthsSource)

        if save_attention_weights == True:
            indicesGreedy, lengthsGreedy, attentionIndices, attentionMatrix = \
                encdec.greedyTrans(corpus.targetVoc.bosIndex, corpus.targetVoc.eosIndex,
                                   lengthsSource, embedding.targetEmbedding,
                                   sourceH, (hn, cn), maxGenLen=maxLen, testMode=True)
            attention_scores_matrix_list.append(attentionMatrix)

        else:
            indicesGreedy, lengthsGreedy, attentionIndices = encdec.greedyTrans(
                corpus.targetVoc.bosIndex, corpus.targetVoc.eosIndex, lengthsSource, embedding.targetEmbedding, sourceH, (hn, cn), maxGenLen=maxLen)
        indicesGreedy = indicesGreedy.cpu()

        # FIXME: Remove this hard coding after debugging.
        # replace_UNK = False
        for i in range(batchSize):
            target_tokens = []
            for k in range(lengthsGreedy[i] - 1):
                index = indicesGreedy.data[i, k]
                if index == corpus.targetVoc.unkIndex:
                    index = attentionIndices[i, k]
                    # Especially for Japanese, just replacing the name does not work well.
                    # Replacing Japanese word using wikipedia title dictionary.
                    if replace_UNK == True and batchData[i].sourceOrigStr[index] in langlinks_dict.keys():
                        linked_en_name = langlinks_dict[batchData[i].sourceOrigStr[index]]
                        f_trans.write(linked_en_name.split()[0] + ' ')
                        target_tokens.append(linked_en_name.split()[0])
                        print("{0} was replaced with {1}".format(
                            batchData[i].sourceOrigStr[index],
                            linked_en_name.split()[0]))
                        # f_trans.write(batchData[i].sourceOrigStr[index] + ' ')
                        # target_tokens.append(
                        #     batchData[i].sourceOrigStr[index])
                    else:
                        f_trans.write(batchData[i].sourceOrigStr[index] + ' ')
                        target_tokens.append(
                            batchData[i].sourceOrigStr[index])
                else:
                    f_trans.write(corpus.targetVoc.tokenList[index].str + ' ')
                    target_tokens.append(
                        corpus.targetVoc.tokenList[index].str)

            f_trans.write('\n')
            trans_result.append(" ".join(target_tokens))

            if trans_mode == False:
                for k in range(lengthsTarget[i] - 1):
                    index = batchInputTarget.data[i, k + 1]
                    if index == corpus.targetVoc.unkIndex:
                        f_gold.write(batchData[i].targetUnkMap[k] + ' ')
                    else:
                        f_gold.write(
                            corpus.targetVoc.tokenList[index].str + ' ')
                f_gold.write('\n')
    f_trans.close()

    if trans_mode == False:
        batchInputTarget = batchInputTarget.cuda()
        batchTarget = batchTarget.cuda()
        inputTarget = embedding.getBatchedTargetEmbedding(batchInputTarget)

        loss = encdec(inputTarget, lengthsTarget, lengthsSource,
                      (hn, cn), sourceH, batchTarget)
        loss = loss.sum()
        devPerp += loss.data[0]

        totalTokenCount += tokenCount
        f_gold.close()
        os.system("./bleu.sh 2> DUMMY")

        return trans_result, devPerp, totalTokenCount

    print(attention_scores_matrix_list)
    return trans_result, attention_scores_matrix_list

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans_source', type=str,
                        help='path to a source file to be translated.')
    # The trnas_target option need to be set if you try to calculate BLEU score.
    # Otherwise, you can just set the same file path as `-trans_source`.
    parser.add_argument('--trans_target', type=str,
                        help='path to a target file to be translated.')
    parser.add_argument('--train_source', type=str,
                        help='path to an archived NMT train source file')
    parser.add_argument('--train_target', type=str,
                        help='path to an archived NMT train target file')
    parser.add_argument('--embedding_params', type=str,
                        help='path to an archived NMT trained embedding model')
    parser.add_argument('--encdec_params', type=str,
                        help='path to an archived NMT trained encdec trained model')
    parser.add_argument('--seed', type=int, default=3,
                        help='seed for NMT params.')
    parser.add_argument('--beam_size', type=int, default=10,
                        help='set beam search window size. If beam size is set to 0, the model would conduct greedy search.')
    parser.add_argument('--use_attenMatrix', action='store_true',
                        help='set true if you try to attention matrix.')
    parser.add_argument('--bleu_score', action='store_true',
                        help='calculate bleu score.')

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    trans_from_files_beam(args.trans_source, args.trans_target,
                          args.train_source, args.train_target,
                          args.embedding_params, args.encdec_params,
                          args.seed, args.beam_size,
                          args.use_attenMatrix)

    # Calculate the bleu score if the bleu score option is set to True.
    if args.bleu_score == True:
        os.system("./bleu.sh 2> DUMMY")
        f_trans = open('./bleu.txt', 'r')
        for line in f_trans:
            devGleu = float(line.split()[2][0:-1])
            break
        f_trans.close()

        devPerp = math.exp(devPerp/totalTokenCount)
        print("Dev perp:", devPerp)
        print("Dev BLEU:", devGleu)

if __name__ == "__main__":
    main()