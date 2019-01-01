
def buildBatchList(dataSize, batchSize):
    batchList = []
    if dataSize % batchSize == 0:
        numBatch = dataSize//batchSize
    else:
        numBatch = int(dataSize/batchSize)+1

    for i in range(numBatch):
        batch = []
        batch.append(i*batchSize)
        if i == numBatch-1:
            batch.append(dataSize-1)
        else:
            batch.append((i+1)*batchSize-1)
        batchList.append(batch)

    return batchList
