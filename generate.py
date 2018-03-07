###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import torch
from torch.autograd import Variable
import data

import util

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model: given prefix, generate next word')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')

args = parser.parse_args()

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model_saved = torch.load(f)

model = model_saved['model']
encoder = model_saved['encoder']
decoder = model_saved['decoder']

model.eval()
encoder.eval()
decoder.eval()

model.cpu()
encoder.cpu()
decoder.cpu()

corpus_raw = data.Corpus(args.data)
word_rank = util.word_rank_dictionary(corpus = corpus_raw)
corpus = util.Rand_Idxed_Corpus(corpus = corpus_raw, word_rank = word_rank)


#corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)

with open(args.outf, 'wt+') as outf:
    for i in range(args.words):
        emb = encoder(input)
        output, hidden = model(emb, hidden)
        logits = decoder.full(output.view(-1, output.size(2)))

        word_weights = logits.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]
        outf.write(word + ('\n' if i % 20 == 19 else ' '))
