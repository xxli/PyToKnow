import json
import os
import sys
import re

import datasplit

class AI2:
    ''' class for AI2 dataset

    '''

    def __init__(self):
        self.index = 0
        self.question = ""
        self.equations = []
        self.solutions = []

        self.questionTemplate = ""
        self.equationsTemplate = []
        self.numberVariableDict = {}

    def generateTemplage(self):
        self.getQuestionTemplate()
        self.getEquationTemplate()

    def getQuestionTemplate(self):
        '''generate question template from original question,
        including changing the number into number[x] format.
        '''
        words = self.question.split(" ")
        start = 0
        index = 0
        for word in words:
            if is_numeric(word): 
                variable = "number" + str(index)
                self.numberVariableDict[float(word)] = variable
                index += 1
                words[start] = variable
            start += 1
        self.questionTemplate = " ".join(words)
        
    def getEquationTemplate(self):
        ''' replace the numbers in equation to variables according to
        given numberVariableDict
        '''
        for equation in self.equations:
            words = equation.split(" ")
            start = 0
            for word in words:
                if is_numeric(word):
                    if self.numberVariableDict[float(word)] != None:
                        words[start] = self.numberVariableDict[float(word)]
                    else:
                        raise Exception("equation " + equation + \
                            " contains numbers which are not indexed")        
                if word == 'x':
                    words[start] = "numberx"
                start += 1
            equationTemplate =  " ".join(words)
            self.equationsTemplate.append(equationTemplate)

def is_numeric(s):
    '''Determine whether the str matches a integer or a float number.
    '''
    value = re.compile(r'^[-+]?\d*\.{0,1}\d+$')
    result=value.match(s)
    if result:
        return True
    else:
        return False

def readAI2_list_from_file(filename, isTrain=True, isLower=True):
    '''read AI2 dataset

    return: mwplist
    '''
    mwplist = list()
    with open(filename,'r') as file:
        allQuestions = file.read()
        if not isLower:
            allQuestions = allQuestions.lower()
        allQuestionJson = json.loads(allQuestions)
        for oneQuestionJson in allQuestionJson:
            mwpone = AI2()
            mwpone.index = oneQuestionJson['iIndex']
            mwpone.question = oneQuestionJson['sQuestion']
            if isTrain:
                mwpone.equations = oneQuestionJson['lEquations']
                mwpone.solutions= oneQuestionJson['lSolutions']
            mwplist.append(mwpone)
    return  mwplist

def readAI2_tra(filename, isTrain=True, isLower=True):
    '''read Ai2 dataset

    return: three lists, separated as questions, equations and solutions.
    '''
    mwplist = list()
    with open(filename,'r') as file:
        allQuestions = file.read()
        if not isLower:
            allQuestions = allQuestions.lower()
        allQuestionJson = json.loads(allQuestions)
        for oneQuestionJson in allQuestionJson:
            mwpone = AI2()
            mwpone.index = oneQuestionJson['iIndex']
            mwpone.question = oneQuestionJson['sQuestion']
            if isTrain:
                mwpone.equations = oneQuestionJson['lEquations']
                mwpone.solutions= oneQuestionJson['lSolutions']
    return  mwplist


def outputOpenNMT(pairlist, filename):
    ''' output mwp list to files

    args:
        mwpList = [[1train, 1test], [2train, 2test], ...]
    '''
    assert isinstance(pairlist, list)
    start = 0
    for mwp in pairlist:
        trainDataset = mwp[0]
        testDataset = mwp[1]
        trainSrcFileName = filename+"-"+str(start)+"-train-src"
        trainTmptFileName = filename+"-"+str(start)+"-train-tmpt"
        trainSolFileName = filename+"-"+str(start)+"-train-sol"
        testSrcFileName = filename+"-"+str(start)+"-test-src"
        testTmptFileName = filename+"-"+str(start)+"-test-tmpt"
        testSolFileName = filename+"-"+str(start)+"-test-sol"
        trainSrcFile = open(trainSrcFileName, 'w')
        trainTmptFile = open(trainTmptFileName, 'w')
        trainSolFile = open(trainSolFileName, 'w')
        for data in trainDataset:
            question = data.questionTemplate
            trainSrcFile.write(question+"\n")
            equation = ",".join(data.equationsTemplate)
            trainTmptFile.write(equation+"\n")
            solution = ",".join(data.solutions)
            trainSolFile.write(solution+"\n")
        trainSrcFile.close()
        trainTmptFile.close()
        testSrcFile = open(testSrcFileName, 'w')
        testTmptFile = open(testTmptFileName, 'w')
        testSolFile = open(testSolFileName, 'w')
        for data in testDataset:
            question = data.questionTemplate
            testSrcFile.write(question+"\n")
            equation = ", ".join(data.equationsTemplate)
            testTmptFile.write(equation+"\n")
            solution = ",".join(data.solutions)
            testSolFile.write(solution+"\n")
        testSrcFile.close()
        testTmptFile.close()
        start += 1

def generateDataset(datasetType, filename, outputType, outputFilename):
    ''' generate the training and test data from original json file
    '''
    if datasetType == "AI2":
        mwplist = readAI2(filename)
        for mwp in mwplist:
            mwp.generateTemplage()
        mwpcvlist = datasplit.crossValidation(mwplist, item=3)
    if outputType == "openNMT":
        outputOpenNMT(mwpcvlist, outputFilename)



if __name__=="__main__":
    dataset_name = "AI2"
    AI2Filename = r"D:\dataset\MaWPS\AddSub.json"
    AI2OutputFilename = r"D:\experiments\AI2\AI2"
    generateDataset(sys.argv[1], sys.argv[2], sys.argv[3])