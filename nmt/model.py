import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Embedding(nn.Module):

    def __init__(self, sourceEmbedDim, targetEmbedDim, sourceVocSize, targetVocSize):
        super(Embedding, self).__init__()

        self.sourceEmbedding = nn.Embedding(sourceVocSize, sourceEmbedDim)
        self.targetEmbedding = nn.Embedding(targetVocSize, targetEmbedDim)

        self.initWeights()

    def initWeights(self):
        initScale = 0.1

        self.sourceEmbedding.weight.data.uniform_(-initScale, initScale)
        self.targetEmbedding.weight.data.uniform_(-initScale, initScale)

    def getBatchedSourceEmbedding(self, batchInput):
        return self.sourceEmbedding(batchInput)

    def getBatchedTargetEmbedding(self, batchInput):
        return self.targetEmbedding(batchInput)


class WordPredictor(nn.Module):

    def __init__(self, inputDim, outputDim, ignoreIndex=-1):
        super(WordPredictor, self).__init__()

        self.softmaxLayer = nn.Linear(inputDim, outputDim)
        self.loss = nn.CrossEntropyLoss(
            ignore_index=ignoreIndex)

        self.initWeight()

    def initWeight(self):
        self.softmaxLayer.weight.data.zero_()
        self.softmaxLayer.bias.data.zero_()

    def forward(self, input, target=None):
        output = self.softmaxLayer(input)
        if target is not None:
            return self.loss(output, target)
        else:
            return output


class DecCand:
    def __init__(self, score_=0.0, fin_=False, sentence_=[], attenIndex_=[], attenMatrix_=[]):
        self.score = score_
        self.fin = fin_
        self.sentence = sentence_
        self.attenIndex = attenIndex_
        self.attenMatrix = attenMatrix_


class EncDec(nn.Module):

    def __init__(self, sourceEmbedDim, targetEmbedDim, hiddenDim, targetVocSize, dropoutRate=0.2, numLayers=1):
        super(EncDec, self).__init__()

        self.numLayers = numLayers
        self.dropout = nn.Dropout(p=dropoutRate)

        self.encoder = nn.LSTM(input_size=sourceEmbedDim, hidden_size=hiddenDim,
                               num_layers=self.numLayers, dropout=0.0, bidirectional=True)

        self.decoder = nn.LSTM(input_size=targetEmbedDim + hiddenDim, hidden_size=hiddenDim,
                               num_layers=self.numLayers, dropout=0.0, bidirectional=False, batch_first=True)

        self.attentionLayer = nn.Linear(2 * hiddenDim, hiddenDim, bias=False)
        self.finalHiddenLayer = nn.Linear(3 * hiddenDim, targetEmbedDim)
        self.finalHiddenAct = nn.Tanh()

        self.wordPredictor = WordPredictor(targetEmbedDim, targetVocSize)

        self.targetEmbedDim = targetEmbedDim
        self.hiddenDim = hiddenDim

        self.initWeight()

    def initWeight(self):
        initScale = 0.1

        self.encoder.weight_ih_l0.data.uniform_(-initScale, initScale)
        self.encoder.weight_hh_l0.data.uniform_(-initScale, initScale)
        self.encoder.bias_ih_l0.data.zero_()
        self.encoder.bias_hh_l0.data.zero_()
        self.encoder.bias_hh_l0.data[self.hiddenDim:2 *
                                     self.hiddenDim].fill_(1.0)  # forget bias = 1

        self.encoder.weight_ih_l0_reverse.data.uniform_(-initScale, initScale)
        self.encoder.weight_hh_l0_reverse.data.uniform_(-initScale, initScale)
        self.encoder.bias_ih_l0_reverse.data.zero_()
        self.encoder.bias_hh_l0_reverse.data.zero_()
        # forget bias = 1
        self.encoder.bias_hh_l0_reverse.data[self.hiddenDim:2 *
                                             self.hiddenDim].fill_(1.0)

        self.decoder.weight_ih_l0.data.uniform_(-initScale, initScale)
        self.decoder.weight_hh_l0.data.uniform_(-initScale, initScale)
        self.decoder.bias_ih_l0.data.zero_()
        self.decoder.bias_hh_l0.data.zero_()
        self.decoder.bias_hh_l0.data[self.hiddenDim:2 *
                                     self.hiddenDim].fill_(1.0)  # forget bias = 1

        if self.numLayers == 2:
            self.encoder.weight_ih_l1.data.uniform_(-initScale, initScale)
            self.encoder.weight_hh_l1.data.uniform_(-initScale, initScale)
            self.encoder.bias_ih_l1.data.zero_()
            self.encoder.bias_hh_l1.data.zero_()
            # forget bias = 1
            self.encoder.bias_hh_l1.data[self.hiddenDim:2 *
                                         self.hiddenDim].fill_(1.0)

            self.encoder.weight_ih_l1_reverse.data.uniform_(
                -initScale, initScale)
            self.encoder.weight_hh_l1_reverse.data.uniform_(
                -initScale, initScale)
            self.encoder.bias_ih_l1_reverse.data.zero_()
            self.encoder.bias_hh_l1_reverse.data.zero_()
            # forget bias = 1
            self.encoder.bias_hh_l1_reverse.data[self.hiddenDim:2 *
                                                 self.hiddenDim].fill_(1.0)

            self.decoder.weight_ih_l1.data.uniform_(-initScale, initScale)
            self.decoder.weight_hh_l1.data.uniform_(-initScale, initScale)
            self.decoder.bias_ih_l1.data.zero_()
            self.decoder.bias_hh_l1.data.zero_()
            # forget bias = 1
            self.decoder.bias_hh_l1.data[self.hiddenDim:2 *
                                         self.hiddenDim].fill_(1.0)

        self.attentionLayer.weight.data.zero_()

        self.finalHiddenLayer.weight.data.uniform_(-initScale, initScale)
        self.finalHiddenLayer.bias.data.zero_()

    def encode(self, inputSource, lengthsSource):
        packedInput = nn.utils.rnn.pack_padded_sequence(
            inputSource, lengthsSource, batch_first=True)

        # hn, ch: (layers*direction, B, Ds)
        h, (hn, cn) = self.encoder(packedInput)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        if self.numLayers == 1:
            hn = (hn[0] + hn[1]).unsqueeze(0)
            cn = (cn[0] + cn[1]).unsqueeze(0)
        else:
            hn0 = (hn[0] + hn[1]).unsqueeze(0)
            cn0 = (cn[0] + cn[1]).unsqueeze(0)
            hn1 = (hn[2] + hn[3]).unsqueeze(0)
            cn1 = (cn[2] + cn[3]).unsqueeze(0)

            hn = torch.cat((hn0, hn1), dim=0)
            cn = torch.cat((cn0, cn1), dim=0)

        return h, (hn, cn)

    def forward(self, inputTarget, lengthsTarget, lengthsSource, hidden0Target, sourceH, target=None):
        batchSize = sourceH.size(0)
        maxLen = lengthsTarget[0]

        for i in range(batchSize):
            maxLen = max(maxLen, lengthsTarget[i])

        finalHidden = inputTarget.new(batchSize, maxLen, self.targetEmbedDim)
        prevFinalHidden = inputTarget.new(
            batchSize, 1, self.targetEmbedDim).zero_()

        newShape = sourceH.size(0), sourceH.size(
            1), self.hiddenDim  # (B, Ls, Dt)
        sourceHtrans = sourceH.contiguous().view(sourceH.size(
            0) * sourceH.size(1), sourceH.size(2))  # (B*Ls, Ds)
        sourceHtrans = self.attentionLayer(sourceHtrans)  # (B*Ls, Dt)
        sourceHtrans = sourceHtrans.view(
            *newShape).transpose(1, 2)  # (B, Dt, Ls)

        for i in range(maxLen):
            hi, hidden0Target = self.decoder(torch.cat((inputTarget[:, i, :].unsqueeze(
                1), prevFinalHidden), dim=2), hidden0Target)  # hi: (B, 1, Dt)

            if self.numLayers != 1:  # residual connection for this decoder
                hi = hidden0Target[0][0] + hidden0Target[0][1]
                hi = hi.unsqueeze(1)

            attentionScores_ = torch.bmm(
                hi, sourceHtrans).transpose(1, 2)  # (B, Ls, 1)

            attentionScores = attentionScores_.new(
                attentionScores_.size()).fill_(-1024.0)
            for j in range(batchSize):
                attentionScores[j, :lengthsSource[j]].zero_()
            attentionScores += attentionScores_

            attentionScores = attentionScores.transpose(1, 2)  # (B, 1, Ls)
            attentionScores = F.softmax(attentionScores, dim=2)

            contextVec = torch.bmm(attentionScores, sourceH)  # (B, 1, Ds)

            prevFinalHidden = torch.cat(
                (hi, contextVec), dim=2)  # (B, 1, Ds+Dt)
            prevFinalHidden = self.dropout(prevFinalHidden)
            prevFinalHidden = self.finalHiddenLayer(prevFinalHidden)
            prevFinalHidden = self.finalHiddenAct(prevFinalHidden)
            prevFinalHidden = self.dropout(prevFinalHidden)

            finalHidden[:, i, :] = prevFinalHidden.squeeze(1)

        finalHidden = finalHidden.contiguous().view(
            finalHidden.size(0) * finalHidden.size(1), finalHidden.size(2))
        output = self.wordPredictor(finalHidden, target)

        return output

    def greedyTrans(self, bosIndex, eosIndex, lengthsSource, targetEmbedding, sourceH, hidden0Target, device, maxGenLen=100):
        batchSize = sourceH.size(0)
        i = 1
        eosCount = 0
        targetWordIndices = torch.LongTensor(
            batchSize, maxGenLen).fill_(bosIndex).to(device)
        attentionIndices = targetWordIndices.new(targetWordIndices.size())
        targetWordLengths = torch.LongTensor(batchSize).fill_(0)
        fin = [False] * batchSize

        newShape = sourceH.size(0), sourceH.size(
            1), hidden0Target[0].size(2)  # (B, Ls, Dt)
        sourceHtrans = sourceH.contiguous().view(sourceH.size(
            0) * sourceH.size(1), sourceH.size(2))  # (B*Ls, Ds)
        sourceHtrans = self.attentionLayer(sourceHtrans)  # (B*Ls, Dt)
        sourceHtrans = sourceHtrans.view(
            *newShape).transpose(1, 2)  # (B, Dt, Ls)

        prevFinalHidden = sourceH.new(
            batchSize, 1, self.targetEmbedDim).zero_()

        while (i < maxGenLen) and (eosCount < batchSize):
            inputTarget = targetEmbedding(
                targetWordIndices[:, i - 1].unsqueeze(1))
            hi, hidden0Target = self.decoder(torch.cat(
                (inputTarget, prevFinalHidden), dim=2), hidden0Target)  # hi: (B, 1, Dt)

            if self.numLayers != 1:
                hi = hidden0Target[0][0] + hidden0Target[0][1]
                hi = hi.unsqueeze(1)

            attentionScores_ = torch.bmm(
                hi, sourceHtrans).transpose(1, 2)  # (B, Ls, 1)

            attentionScores = attentionScores_.new(
                attentionScores_.size()).fill_(-1024.0)
            for j in range(batchSize):
                attentionScores[j, :lengthsSource[j]].zero_()
            attentionScores += attentionScores_

            attentionScores = attentionScores.transpose(1, 2)  # (B, 1, Ls)
            attentionScores = F.softmax(attentionScores, dim=2)

            attnProb, attnIndex = torch.max(attentionScores, dim=2)
            for j in range(batchSize):
                attentionIndices[j, i - 1] = attnIndex.data[j, 0]

            contextVec = torch.bmm(attentionScores, sourceH)  # (B, 1, Ds)
            finalHidden = torch.cat((hi, contextVec), 2)  # (B, 1, Dt+Ds)
            finalHidden = self.dropout(finalHidden)
            finalHidden = self.finalHiddenLayer(finalHidden)
            finalHidden = self.finalHiddenAct(finalHidden)
            finalHidden = self.dropout(finalHidden)
            prevFinalHidden = finalHidden  # (B, 1, Dt)

            finalHidden = finalHidden.contiguous().view(
                finalHidden.size(0) * finalHidden.size(1), finalHidden.size(2))
            output = self.wordPredictor(finalHidden)

            maxProb, sampledIndex = torch.max(output, dim=1)
            targetWordIndices[:, i].copy_(sampledIndex)

            for j in range(batchSize):
                if not fin[j] and targetWordIndices[j, i - 1].item() != eosIndex:
                    targetWordLengths[j] += 1
                    if sampledIndex[j].item() == eosIndex:
                        eosCount += 1
                        fin[j] = True

            i += 1

        targetWordIndices = targetWordIndices[:, 1:i]  # i-1: no EOS

        return targetWordIndices, list(targetWordLengths), attentionIndices

    def beamSearch(self, bosIndex, eosIndex, lengthsSource, targetEmbedding, sourceH, hidden0Target, device, beamSize=1, penalty=0.75, maxGenLen=100):
        batchSize = sourceH.size(0)

        targetWordIndices = torch.LongTensor(
            batchSize, maxGenLen).fill_(bosIndex).to(device)
        attentionIndices = targetWordIndices.new(targetWordIndices.size())
        targetWordLengths = torch.LongTensor(batchSize).fill_(0)

        newShape = sourceH.size(0), sourceH.size(
            1), hidden0Target[0].size(2)  # (B, Ls, Dt)
        sourceHtrans = sourceH.contiguous().view(sourceH.size(
            0) * sourceH.size(1), sourceH.size(2))  # (B*Ls, Ds)
        sourceHtrans = self.attentionLayer(sourceHtrans)  # (B*Ls, Dt)
        sourceHtrans = sourceHtrans.view(
            *newShape).transpose(1, 2)  # (B, Dt, Ls)

        sourceHtrans_ = sourceHtrans.new(
            beamSize, sourceHtrans.size(1), sourceHtrans.size(2))
        sourceH_ = sourceH.new(beamSize, sourceH.size(1), sourceH.size(2))
        prevFinalHidden = sourceH.new(beamSize, 1, self.targetEmbedDim).zero_()

        sampledIndex = torch.LongTensor(beamSize).zero_()

        h0 = hidden0Target[0]
        c0 = hidden0Target[1]
        h0_ = h0.data.new(h0.size(0), beamSize, h0.size(2))
        c0_ = c0.data.new(c0.size(0), beamSize, c0.size(2))

        for dataIndex in range(batchSize):
            i = 1
            prevFinalHidden.zero_()
            sourceHtrans_.zero_()
            sourceHtrans_ += sourceHtrans[dataIndex]
            sourceH_.zero_()
            sourceH_ += sourceH[dataIndex]
            h0_.zero_()
            c0_.zero_()
            h0_ += h0[:, dataIndex, :].unsqueeze(1)
            c0_ += c0[:, dataIndex, :].unsqueeze(1)
            hidden0Target_ = (h0_, c0_)

            cand = []
            for j in range(beamSize):
                cand.append(DecCand(sentence_=[bosIndex]))

            while i < maxGenLen and not cand[0].fin:
                index = []
                for j in range(beamSize):
                    index.append([cand[j].sentence[-1]])
                index = torch.LongTensor(index).to(device)
                inputTarget = targetEmbedding(index)

                hi, hidden0Target_ = self.decoder(torch.cat(
                    (inputTarget, prevFinalHidden), dim=2), hidden0Target_)  # hi: (B, 1, Dt)

                if self.numLayers != 1:
                    hi = hidden0Target_[0][0] + hidden0Target_[0][1]
                    hi = hi.unsqueeze(1)

                attentionScores_ = torch.bmm(
                    hi, sourceHtrans_).transpose(1, 2)  # (B, Ls, 1)

                attentionScores = attentionScores_.new(
                    attentionScores_.size()).fill_(-1024.0)
                attentionScores[:, :lengthsSource[dataIndex]].zero_()
                attentionScores += attentionScores_

                attentionScores = attentionScores.transpose(1, 2)  # (B, 1, Ls)
                attentionScores = F.softmax(attentionScores, dim=2)

                attnProb, attnIndex = torch.max(attentionScores, dim=2)

                contextVec = torch.bmm(attentionScores, sourceH_)  # (B, 1, Ds)
                finalHidden = torch.cat((hi, contextVec), 2)  # (B, 1, Dt+Ds)
                finalHidden = self.dropout(finalHidden)
                finalHidden = self.finalHiddenLayer(finalHidden)
                finalHidden = self.finalHiddenAct(finalHidden)
                finalHidden = self.dropout(finalHidden)
                prevFinalHidden = finalHidden  # (B, 1, Dt)

                finalHidden = finalHidden.contiguous().view(
                    finalHidden.size(0) * finalHidden.size(1), finalHidden.size(2))
                output = self.wordPredictor(finalHidden)

                output = F.log_softmax(output, dim=1) + penalty

                for j in range(beamSize):
                    if cand[j].fin:
                        output.data[j].fill_(cand[j].score)
                    else:
                        output.data[j] += cand[j].score

                updatedCand = []
                updatedPrevFinalHidden = prevFinalHidden.new(
                    prevFinalHidden.size()).zero_()
                updatedH0 = h0_.new(h0_.size()).zero_()
                updatedC0 = c0_.new(c0_.size()).zero_()

                for j in range(beamSize):
                    maxScore, maxIndex = torch.topk(
                        output.view(output.size(0) * output.size(1)), k=1)

                    row = maxIndex[0].item() // output.size(1)
                    col = maxIndex[0].item() % output.size(1)
                    score = maxScore[0].item()
                    sampledIndex[j] = col

                    if cand[row].fin:
                        updatedCand.append(
                            DecCand(score, True, cand[row].sentence, cand[row].attenIndex))
                        output.data[row].fill_(-1024.0)
                        continue

                    updatedCand.append(DecCand(
                        score, False, cand[row].sentence + [], cand[row].attenIndex + [attnIndex[row, 0].item()]))
                    updatedPrevFinalHidden[j] = prevFinalHidden[row]
                    updatedH0[:, j, :] = hidden0Target_[
                        0][:, row, :].unsqueeze(1)
                    updatedC0[:, j, :] = hidden0Target_[
                        1][:, row, :].unsqueeze(1)

                    if i == 1:
                        output[:, col].fill_(-1024.0)
                    else:
                        output[row, col] = -1024.0

                for j in range(beamSize):
                    if updatedCand[j].fin:
                        continue

                    if sampledIndex[j] == eosIndex:
                        updatedCand[j].fin = True

                    updatedCand[j].sentence.append(sampledIndex[j].item())

                # cand = sorted(updatedCand, key = lambda x:
                # -x.score/len(x.sentence))
                cand = updatedCand
                prevFinalHidden = updatedPrevFinalHidden
                hidden0Target_ = (updatedH0, updatedC0)
                i += 1

            targetWordLengths[dataIndex] = len(cand[0].sentence) - 1
            for j in range(targetWordLengths[dataIndex]):
                targetWordIndices[dataIndex, j] = cand[0].sentence[j]
                attentionIndices[dataIndex, j] = cand[0].attenIndex[j]

        return targetWordIndices[:, 1:], list(targetWordLengths), attentionIndices

    def beamSearchAttn(self, bosIndex, eosIndex, lengthsSource, targetEmbedding, sourceH, hidden0Target, device, beamSize=1, penalty=0.75, maxGenLen=100):
        batchSize = sourceH.size(0)

        targetWordIndices = torch.LongTensor(
            batchSize, maxGenLen).fill_(bosIndex).to(device)
        attentionIndices = targetWordIndices.new(targetWordIndices.size())
        targetWordLengths = torch.LongTensor(batchSize).fill_(0)

        newShape = sourceH.size(0), sourceH.size(
            1), hidden0Target[0].size(2)  # (B, Ls, Dt)
        sourceHtrans = sourceH.contiguous().view(sourceH.size(
            0) * sourceH.size(1), sourceH.size(2))  # (B*Ls, Ds)
        sourceHtrans = self.attentionLayer(sourceHtrans)  # (B*Ls, Dt)
        sourceHtrans = sourceHtrans.view(
            *newShape).transpose(1, 2)  # (B, Dt, Ls)

        sourceHtrans_ = sourceHtrans.new(
            beamSize, sourceHtrans.size(1), sourceHtrans.size(2))
        sourceH_ = sourceH.new(beamSize, sourceH.size(1), sourceH.size(2))
        prevFinalHidden = sourceH.new(beamSize, 1, self.targetEmbedDim).zero_()

        sampledIndex = torch.LongTensor(beamSize).zero_()

        h0 = hidden0Target[0]
        c0 = hidden0Target[1]
        h0_ = h0.data.new(h0.size(0), beamSize, h0.size(2))
        c0_ = c0.data.new(c0.size(0), beamSize, c0.size(2))

        for dataIndex in range(batchSize):
            i = 1
            prevFinalHidden.zero_()
            sourceHtrans_.zero_()
            sourceHtrans_ += sourceHtrans[dataIndex]
            sourceH_.zero_()
            sourceH_ += sourceH[dataIndex]
            h0_.zero_()
            c0_.zero_()
            h0_ += h0[:, dataIndex, :].unsqueeze(1)
            c0_ += c0[:, dataIndex, :].unsqueeze(1)
            hidden0Target_ = (h0_, c0_)

            cand = []
            for j in range(beamSize):
                cand.append(DecCand(sentence_=[bosIndex]))

            while i < maxGenLen and not cand[0].fin:
                index = []
                for j in range(beamSize):
                    index.append([cand[j].sentence[-1]])
                index = torch.LongTensor(index).to(device)
                inputTarget = targetEmbedding(index)

                hi, hidden0Target_ = self.decoder(torch.cat(
                    (inputTarget, prevFinalHidden), dim=2), hidden0Target_)  # hi: (B, 1, Dt)

                if self.numLayers != 1:
                    hi = hidden0Target_[0][0] + hidden0Target_[0][1]
                    hi = hi.unsqueeze(1)

                attentionScores_ = torch.bmm(
                    hi, sourceHtrans_).transpose(1, 2)  # (B, Ls, 1)

                attentionScores = attentionScores_.new(
                    attentionScores_.size()).fill_(-1024.0)
                attentionScores[:, :lengthsSource[dataIndex]].zero_()
                attentionScores += attentionScores_

                attentionScores = attentionScores.transpose(1, 2)  # (B, 1, Ls)
                attentionScores = F.softmax(attentionScores, dim=2)

                attnProb, attnIndex = torch.max(attentionScores, dim=2)

                contextVec = torch.bmm(attentionScores, sourceH_)  # (B, 1, Ds)
                finalHidden = torch.cat((hi, contextVec), 2)  # (B, 1, Dt+Ds)
                finalHidden = self.dropout(finalHidden)
                finalHidden = self.finalHiddenLayer(finalHidden)
                finalHidden = self.finalHiddenAct(finalHidden)
                finalHidden = self.dropout(finalHidden)
                prevFinalHidden = finalHidden  # (B, 1, Dt)

                finalHidden = finalHidden.contiguous().view(
                    finalHidden.size(0) * finalHidden.size(1), finalHidden.size(2))
                output = self.wordPredictor(finalHidden)

                output = F.log_softmax(output, dim=1) + penalty

                for j in range(beamSize):
                    if cand[j].fin:
                        output.data[j].fill_(cand[j].score)
                    else:
                        output.data[j] += cand[j].score

                updatedCand = []
                updatedPrevFinalHidden = prevFinalHidden.new(
                    prevFinalHidden.size()).zero_()
                updatedH0 = h0_.new(h0_.size()).zero_()
                updatedC0 = c0_.new(c0_.size()).zero_()

                for j in range(beamSize):
                    maxScore, maxIndex = torch.topk(
                        output.view(output.size(0) * output.size(1)), k=1)

                    row = maxIndex[0].item() // output.size(1)
                    col = maxIndex[0].item() % output.size(1)
                    score = maxScore[0].item()
                    sampledIndex[j] = col

                    if cand[row].fin:
                        updatedCand.append(
                            DecCand(score, True, cand[row].sentence, cand[row].attenIndex, cand[row].attenMatrix))
                        output.data[row].fill_(-1024.0)
                        continue

                    # updatedCand.append(DecCand(
                    # score, False, cand[row].sentence + [],
                    # cand[row].attenIndex + [attnIndex[row, 0].item()])

                    updatedCand.append(DecCand(
                        score, False, cand[row].sentence + [],
                        cand[row].attenIndex + [attnIndex[row, 0].item()],
                        cand[row].attenMatrix + [list(np.array(attentionScores[0][0].data))]))

                    updatedPrevFinalHidden[j] = prevFinalHidden[row]
                    updatedH0[:, j, :] = hidden0Target_[
                        0][:, row, :].unsqueeze(1)
                    updatedC0[:, j, :] = hidden0Target_[
                        1][:, row, :].unsqueeze(1)

                    if i == 1:
                        output[:, col].fill_(-1024.0)
                    else:
                        output[row, col] = -1024.0

                for j in range(beamSize):
                    if updatedCand[j].fin:
                        continue

                    if sampledIndex[j] == eosIndex:
                        updatedCand[j].fin = True

                    updatedCand[j].sentence.append(sampledIndex[j].item())

                # cand = sorted(updatedCand, key = lambda x:
                # -x.score/len(x.sentence))
                cand = updatedCand
                prevFinalHidden = updatedPrevFinalHidden
                hidden0Target_ = (updatedH0, updatedC0)
                i += 1

            targetWordLengths[dataIndex] = len(cand[0].sentence) - 1
            attentionMatrix = []
            for j in range(targetWordLengths[dataIndex]):
                targetWordIndices[dataIndex, j] = cand[0].sentence[j]
                attentionIndices[dataIndex, j] = cand[0].attenIndex[j]
                # attentionMatrix[dataIndex, j] = cand[0].attenMatrix[j]

        return targetWordIndices[:, 1:], list(targetWordLengths), attentionIndices, cand[0].attenMatrix
