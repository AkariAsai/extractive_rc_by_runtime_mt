import os
import torch

from torch.autograd import Variable


class Token:
    def __init__(self, str_='', count_=0):
        self.str = str_
        self.count = count_


class Vocabulary:
    def __init__(self):
        self.UNK = 'UNK'  # unkown words
        self.EOS = '<EOS>'  # the end-of-sequence token
        self.BOS = '<BOS>'  # the beginning-of-sequence token
        self.PAD = '<PAD>'  # padding

        self.unkIndex = -1
        self.eosIndex = -1
        self.bosIndex = -1
        self.padIndex = -1

        self.tokenIndex = {}
        self.tokenList = []

    def getTokenIndex(self, str):
        if str in self.tokenIndex:
            return self.tokenIndex[str]
        else:
            return self.unkIndex

    def add(self, str, count):
        if str not in self.tokenIndex:
            self.tokenList.append(Token(str, count))
            self.tokenIndex[str] = len(self.tokenList)-1

    def size(self):
        return len(self.tokenList)

    def outputTokenList(self, fileName):
        f = open(fileName, 'w')
        for t in self.tokenList:
            f.write(t.str + '\n')
        f.close()


class Data:
    def __init__(self, sourceText_, targetText_, sourceOrigStr_=None, targetUnkMap_=None):
        self.sourceText = sourceText_
        self.sourceOrigStr = sourceOrigStr_

        self.targetText = targetText_
        self.targetUnkMap = targetUnkMap_


class Corpus:
    def __init__(self, sourceTrainFile='', sourceOrigTrainFile='', targetTrainFile='', sourceDevFile='', sourceOrigDevFile='', targetDevFile='', minFreqSource=1, minFreqTarget=1, maxTokenLen=100000):
        self.sourceVoc = Vocabulary()
        self.targetVoc = Vocabulary()

        # , maxLen = maxTokenLen)
        self.buildVoc(sourceTrainFile, minFreqSource, source=True)
        # , maxLen = maxTokenLen)
        self.buildVoc(targetTrainFile, minFreqTarget, source=False)

        self.trainData = self.buildDataset(
            sourceTrainFile, sourceOrigTrainFile, targetTrainFile, train=True, maxLen=maxTokenLen)
        self.devData = self.buildDataset(
            sourceDevFile, sourceOrigDevFile, targetDevFile, train=False)

    def buildVoc(self, fileName, minFreq, source, maxLen=100000):
        assert os.path.exists(fileName)

        if source:
            voc = self.sourceVoc
        else:
            voc = self.targetVoc

        with open(fileName, 'r') as f:
            tokenCount = {}
            unkCount = 0
            eosCount = 0

            for line in f:
                tokens = line.split()  # w1 w2 ... \n

                if len(tokens) > maxLen:
                    continue

                eosCount += 1

                for t in tokens:
                    if t in tokenCount:
                        tokenCount[t] += 1
                    else:
                        tokenCount[t] = 1

            tokenList = sorted(tokenCount.items(),
                               key=lambda x: -x[1])  # sort by value

            for t in tokenList:
                if t[1] >= minFreq:
                    voc.add(t[0], t[1])
                else:
                    unkCount += t[1]

            '''
            Add special tokens
            '''
            voc.add(voc.UNK, unkCount)
            voc.add(voc.BOS, eosCount)
            voc.add(voc.EOS, eosCount)
            voc.add(voc.PAD, 0)

            voc.unkIndex = voc.getTokenIndex(voc.UNK)
            voc.bosIndex = voc.getTokenIndex(voc.BOS)
            voc.eosIndex = voc.getTokenIndex(voc.EOS)
            voc.padIndex = voc.getTokenIndex(voc.PAD)

    def buildDataset(self, sourceFileName, sourceOrigFileName, targetFileName, train, maxLen=100000):
        assert os.path.exists(
            sourceFileName) and os.path.exists(targetFileName)
        assert os.path.exists(sourceOrigFileName)

        with open(sourceFileName, 'r') as fs, open(sourceOrigFileName, 'r') as fsOrig, open(targetFileName, 'r') as ft:
            dataset = []

            for (lineSource, lineSourceOrig, lineTarget) in zip(fs, fsOrig, ft):
                tokensSource = lineSource.split()  # w1 w2 ... \n
                if train:
                    tokensSourceOrig = None
                else:
                    tokensSourceOrig = lineSourceOrig.split()  # w1 w2 ... \n
                tokensTarget = lineTarget.split()  # w1 w2 ... \n

                if len(tokensSource) > maxLen or len(tokensTarget) > maxLen or len(tokensSource) == 0 or len(tokensTarget) == 0:
                    continue

                tokenIndicesSource = torch.LongTensor(len(tokensSource))
                tokenIndicesTarget = torch.LongTensor(len(tokensTarget))
                unkMapTarget = {}

                for i in range(len(tokensSource)):
                    t = tokensSource[i]
                    tokenIndicesSource[i] = self.sourceVoc.getTokenIndex(t)

                for i in range(len(tokensTarget)):
                    t = tokensTarget[i]
                    tokenIndicesTarget[i] = self.targetVoc.getTokenIndex(t)
                    if tokenIndicesTarget[i] == self.targetVoc.unkIndex:
                        unkMapTarget[i] = t

                dataset.append(
                    Data(tokenIndicesSource, tokenIndicesTarget, tokensSourceOrig, unkMapTarget))

        return dataset

    def processBatchInfoNMT(self, batch, train, device):
        begin = batch[0]
        end = batch[1]
        batchSize = end-begin+1

        '''
        Process source info
        '''
        if train:
            data = sorted(self.trainData[begin:end+1],
                          key=lambda x: -len(x.sourceText))
        else:
            data = sorted(self.devData[begin:end+1],
                          key=lambda x: -len(x.sourceText))

        maxLen = len(data[0].sourceText)
        batchInputSource = torch.LongTensor(batchSize, maxLen)
        batchInputSource.fill_(self.sourceVoc.padIndex)
        lengthsSource = []

        for i in range(batchSize):
            l = len(data[i].sourceText)
            lengthsSource.append(l)

            for j in range(l):
                batchInputSource[i, j] = data[i].sourceText[j]

        batchInputSource = batchInputSource.to(device)

        '''
        Process target info
        '''
        data_ = sorted(data, key=lambda x: -len(x.targetText))

        maxLen = len(data_[0].targetText)+1  # for BOS or EOS
        batchInputTarget = torch.LongTensor(batchSize, maxLen)
        batchInputTarget.fill_(self.targetVoc.padIndex)
        lengthsTarget = []
        batchTarget = torch.LongTensor(maxLen*batchSize).fill_(-1)
        targetIndexOffset = 0
        tokenCount = 0.0

        for i in range(batchSize):
            l = len(data[i].targetText)
            lengthsTarget.append(l+1)
            batchInputTarget[i, 0] = self.targetVoc.bosIndex
            for j in range(l):
                batchInputTarget[i, j+1] = data[i].targetText[j]
                batchTarget[targetIndexOffset+j] = data[i].targetText[j]
            batchTarget[targetIndexOffset+l] = self.targetVoc.eosIndex
            targetIndexOffset += maxLen
            tokenCount += (l+1)

        return batchInputSource, lengthsSource, batchInputTarget, batchTarget, lengthsTarget, tokenCount, data
