import sys

def crossValidation(dataset, item=10):
    ''' [item]-cross validation
    arg:
        dataset: list of examples
    
    return: n pairs of training and test dataset, each dataset in every pair \
        is a list of examples. The [[train-1, test-1], [train-2, test-2], ...]
    '''
    
    datasetSize = len(dataset)
    if item < 2:
        print("item must be larger than 1.")
        sys.exit()
    
    patchSize = int(datasetSize/item)

    datasetList = []
    i = 0
    while i < item: 
        datasetList.append([[],[]])
        i += 1
    
    i = 0
    while i < datasetSize:
        data = dataset[i]
        thisPatch = i//patchSize
        j = 0
        while j < item:
            if thisPatch == j:
                datasetList[j][1].append(data)
            else:
                datasetList[j][0].append(data)
            j += 1
        i += 1
    return datasetList