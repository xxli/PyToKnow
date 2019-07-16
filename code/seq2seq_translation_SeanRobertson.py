###
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from data.vocab import Voc
import data.string_util as string_util
import util.time_util as time_util
from model.seq2seq_SeanRobertson import EncoderRNN, AttnDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1

#

def readLangs(filename, reverse=False):
    '''
    To read the data file we will split the file into lines, and then split
    lines into pairs. The files are all English → Other Language, so if we
    want to translate from Other Language → English I added the ``reverse``
    flag to reverse the pairs.
    '''
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(filename, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[string_util.normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Voc instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_voc = Voc("output")
        output_voc = Voc("input")
    else:
        input_voc = Voc("input")
        output_voc = Voc("output")

    return input_voc, output_voc, pairs


def prepareData(filename, reverse=False):
    input_voc, output_voc, pairs = readLangs(filename, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_voc.addSentence(pair[0])
        output_voc.addSentence(pair[1])
    print("Counted words:")
    print(input_voc.name, input_voc.num_words)
    print(output_voc.name, output_voc.num_words)
    return input_voc, output_voc, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]




def trainIter(input_tensor, target_tensor, encoder, decoder, \
    encoder_optimizer, decoder_optimizer, criterion, max_length=10):
    
    encoder_hidden = encoder.initHidden().to(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train(input_voc, output_voc, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [(input_voc.tensorFromSentence(pairs[i][0]).to(device), \
        output_voc.tensorFromSentence(pairs[i][1]).to(device)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = trainIter(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_util.timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    print(plot_losses)
    return encoder, decoder



def evaluate(embedding, encoder, decoder, sentence, max_length=10):
    with torch.no_grad():

        input_tensor = embedding.tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden().to(device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[embedding.input_voc.SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == embedding.output_voc.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(embedding.output_voc.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(embedding, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(embedding, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if __name__ == "__main__":
    path = r"data"
    filename = os.path.join(path, "eng-fra.txt")
    input_voc, output_voc, pairs = prepareData(filename, True)
    print(random.choice(pairs))
    teacher_forcing_ratio = 0.5        
    
    hidden_size = 256
    encoder1 = EncoderRNN(input_voc.num_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_voc.num_words, dropout_p=0.1).to(device)
    train(input_voc, output_voc, encoder1, attn_decoder1, n_iters=5000, print_every=10)
    torch.save(encoder1.state_dict(), os.path.join(path, "encoder.model"))
    torch.save(decoder1.state_dict(), os.path.join(path, "decoder.model"))
    encoder2 = EncoderRNN()
    encoder2.load_state_dict(os.path.join(path, "encoder.model"))
    decoder2 = AttnDecoderRNN()
    decoder2.load_state_dict(os.path.join(path, "decoder.model"))
    evaluateRandomly(voc_to_tensor, encoder2, decoder2)
