import os
import sys
import random
import shutil

import torch
import torch.nn as nn

current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)
from data.vocab import Voc

 


def read_embedding(filename, skip_lines=0):
    ''' Read embedding from stanford glove file

    args:
        skip_lines: the number of lines you want to skip
    return:
        embedding: embedding vector
        Vocab: including 
    '''
    embs = list()
    voc = Voc("word")
    with open(filename, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            l_split = line.strip().split(' ')
            if len(l_split) == 2:
                continue
            embs.append([float(em) for em in l_split[1:]])
            word = l_split[0]
            voc.addWord(word)
    return embs, voc

def add_embedding(input_filename, output_filename, wordlist=[]):
    '''Add new word in input_voc, output_voc, wordlist with its random embedding
    to input embedding file, and output
    '''
    if(os.path.isfile(output_filename)):
        raise Exception(output_filename + " exists. Please check it.")
    shutil.copyfile(input_filename, output_filename)

    embs, embedding_voc = read_embedding(input_filename)
    embedding_dim = len(embs[0])
    embedding_words = embedding_voc.word2index.keys()

    file = open(output_filename, 'a', encoding="utf-8")
    for word in wordlist:
        if word not in embedding_words:
            new_embedding = [str(random.uniform(-1,1)) for i in range(embedding_dim)]
            file.write(word+" "+" ".join(new_embedding)+"\n")
     
    file.close()


def add_embedding_from_file(input_filename, output_filename, filenames=[]):
    '''
    '''    
    wordlist = []
    for filename in filenames:
        voc = Voc()
        with open(filename, 'r') as f:
            for line in f:
                word = line.split()
                voc.addWord(word)
        wordlist.append(voc.word2index.keys())
    wordlist = list(set(wordlist))
    add_embedding(input_filename, output_filename, wordlist)







if __name__ == "__main__":
    add_embedding("D:\\dataset\\glove\\glove.6B.50d.txt", "D:\\experiments\\AI2\\glove.6B.50d.add.txt", \
        ["numberx", "number1", "number2", "number3", "number4", "number5"])