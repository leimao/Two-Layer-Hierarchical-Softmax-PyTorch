import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import data
import math


def word_count(corpus):
    counter = [0] * len(corpus.dictionary.idx2word)
    for i in corpus.train:
        counter[i] += 1
    for i in corpus.valid:
        counter[i] += 1
    for i in corpus.test:
        counter[i] += 1
    return np.array(counter).astype(int)

def word_freq_ordered(corpus):
    # Given a word_freq_rank, we could find the word_idx of that word in the corpus
    counter = word_count(corpus = corpus)
    # idx_order: freq from large to small (from left to right)
    idx_order = np.argsort(-counter)
    return idx_order.astype(int)

def word_rank_dictionary(corpus):
    # Given a word_idx, we could find the frequency rank (0-N, the smaller the rank, the higher frequency the word) of that word in the corpus
    idx_order = word_freq_ordered(corpus = corpus)
    # Reverse
    rank_dictionary = np.zeros(len(idx_order))
    for rank, word_idx in enumerate(idx_order):
        rank_dictionary[word_idx] = rank
    return rank_dictionary.astype(int)



class Rand_Idxed_Corpus(object):
    # Corpus using word rank as index
    def __init__(self, corpus, word_rank):
        self.dictionary = self.convert_dictionary(dictionary = corpus.dictionary, word_rank = word_rank)
        self.train = self.convert_tokens(tokens = corpus.train, word_rank = word_rank)
        self.valid = self.convert_tokens(tokens = corpus.valid, word_rank = word_rank)
        self.test = self.convert_tokens(tokens = corpus.test, word_rank = word_rank)

    def convert_tokens(self, tokens, word_rank):
        rank_tokens = torch.LongTensor(len(tokens))
        for i in range(len(tokens)):
            #print(word_rank[tokens[i]])

            rank_tokens[i] = int(word_rank[tokens[i]])
        return rank_tokens

    def convert_dictionary(self, dictionary, word_rank):
        rank_dictionary = data.Dictionary()
        rank_dictionary.idx2word = [''] * len(dictionary.idx2word)
        for idx, word in enumerate(dictionary.idx2word):
            #print(word_rank)
            #print(rank)
            rank = word_rank[idx]
            rank_dictionary.idx2word[rank] = word
            if word not in rank_dictionary.word2idx:
                rank_dictionary.word2idx[word] = rank
        return rank_dictionary



class Word2VecEncoder(nn.Module):

    def __init__(self, ntoken, ninp, dropout):
        super(Word2VecEncoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        #print(self.encoder.weight[0][0])
        emb = self.encoder(input)
        emb = self.drop(emb)
        return emb

class LinearDecoder(nn.Module):
    def __init__(self, nhid, ntoken):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        decoded = self.decoder(inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2)))
        return decoded.view(inputs.size(0), inputs.size(1), decoded.size(1))


class LogUniformSampler(object):
    def __init__(self, ntokens):

        self.N = ntokens
        self.prob = [0] * self.N

        self.generate_distribution()

    def generate_distribution(self):
        for i in range(self.N):
            self.prob[i] = (np.log(i+2) - np.log(i+1)) / np.log(self.N + 1)

    def probability(idx):
        return self.prob[idx]

    def expected_count(self, num_tries, samples):
        freq = list()
        for sample_idx in samples:
            freq.append(-(np.exp(num_tries * np.log(1-self.prob[sample_idx]))-1))
        return freq

    def accidental_match(self, labels, samples):
        sample_dict = dict()

        for idx in range(len(samples)):
            sample_dict[samples[idx]] = idx

        result = list()
        for idx in range(len(labels)):
            if labels[idx] in sample_dict:
                result.append((idx, sample_dict[labels[idx]]))

        return result

    def sample(self, size, labels):
        log_N = np.log(self.N)

        x = np.random.uniform(low=0.0, high=1.0, size=size)
        value = np.floor(np.exp(x * log_N)).astype(int) - 1
        samples = value.tolist()

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq

    def sample_unique(self, size, labels):
        # Slow. Not Recommended.
        log_N = np.log(self.N)
        samples = list()

        while (len(samples) < size):
            x = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            value = np.floor(np.exp(x * log_N)).astype(int) - 1
            if value in samples:
                continue
            else:
                samples.append(value)

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq



class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens)
        self.params = nn.Linear(nhid, ntokens)

        if tied_weight is not None:
            self.params.weight = tied_weight
        else:
            in_, out_ = self.params.weight.size()
            stdv = math.sqrt(3. / (in_ + out_))
            self.params.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, labels):
        if self.training:
            # sample ids according to word distribution - Unique
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
        else:
            return self.full(inputs)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):

        #print(inputs)
        #print(inputs.size())

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = Variable(torch.LongTensor(sample_ids))
        true_freq = Variable(torch.FloatTensor(true_freq))
        sample_freq = Variable(torch.FloatTensor(sample_freq))

        # gather true labels - weights and frequencies
        true_weights = self.params.weight[labels, :]
        true_bias = self.params.bias[labels]

        # gather sample ids - weights and frequencies
        sample_weights = self.params.weight[sample_ids, :]
        sample_bias = self.params.bias[sample_ids]

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias
        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e37

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(torch.zeros(batch_size).long())
        return logits, new_targets

    def full(self, inputs):
        return self.params(inputs)



class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, ntokens_per_class = None):
        super(HierarchicalSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nhid = nhid

        #self.drop = nn.Dropout(dropout)

        if ntokens_per_class is None:
            ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))

        self.ntokens_per_class = ntokens_per_class

        self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        self.ntokens_actual = self.nclasses * self.ntokens_per_class

        #self.layer_top_W = Variable(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        #self.layer_top_b = Variable(torch.FloatTensor(self.nclasses), requires_grad=True)

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.nclasses), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.nclasses), requires_grad=True)


        #self.layer_bottom_W = Variable(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
        #self.layer_bottom_b = Variable(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)

        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.nclasses, self.nhid, self.ntokens_per_class), requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.nclasses, self.ntokens_per_class), requires_grad=True)


        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

        #self.layer_top_W_check = self.layer_top_W


    def init_weights(self):

        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)


    def forward(self, inputs, labels):

        #print(self.layer_top_W)
        #if (self.layer_top_W_check.data == self.layer_top_W.data):
        #    print(True)
        #else:
        #    print(False)
        #self.layer_top_W_check = self.layer_top_W


        batch_size, d = inputs.size()

        label_position_top = labels / self.ntokens_per_class
        label_position_bottom = labels % self.ntokens_per_class


        layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
        layer_top_probs = self.softmax(layer_top_logits)


        #self.layer_bottom_W[label_position_top]

        #print(torch.squeeze(inputs, dim=1))



        layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + self.layer_bottom_b[label_position_top]
        layer_bottom_probs = self.softmax(layer_bottom_logits)


        target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]

        return target_probs








class HierarchicalSoftmaxMappingModule(nn.Module):
    """
    Build full tree for hierarchical softmax. Not the best option from performance point of view,
    but wasted too much time making it working, so, let's keep it here :)
    """
    def __init__(self):
        super(HierarchicalSoftmaxMappingModule, self).__init__()

    def forward(self, scores, class_indices, cuda=False):
        batch_size, dict_size = scores.size()
        # without initial sigmoid, our predefined values should be large and small vals
        pad_one = Variable(torch.ones(batch_size, 1) * 100.0)
        pad_zer = Variable(torch.ones(batch_size, 1) * (-100.0))
        if cuda:
            pad_one = pad_one.cuda()
            pad_zer = pad_zer.cuda()
        padded_scores = torch.cat([pad_one, pad_zer, scores], dim=1)

        code_len = m.ceil(m.log(dict_size, 2))
        mask = 2**(code_len-1)
        level = 1
        left_indices = []
        right_indices = []
        indices = [2] * batch_size

        while mask > 0:
            left_list = []
            right_list = []
            for batch_idx, right_branch in enumerate(map(lambda v: bool(v & mask), class_indices)):
                cur_index = indices[batch_idx]
                if not right_branch:
                    left_list.append(cur_index)
                    right_list.append(1)
                    cur_index <<= 1
                    cur_index -= 1
                else:
                    left_list.append(0)
                    right_list.append(cur_index)
                    cur_index <<= 1
                indices[batch_idx] = cur_index
            left_indices.append(left_list)
            right_indices.append(right_list)
            level <<= 1
            mask >>= 1

        left_t = Variable(torch.LongTensor(left_indices)).t()
        right_t = Variable(torch.LongTensor(right_indices)).t()
        if cuda:
            left_t = left_t.cuda()
            right_t = right_t.cuda()

        left_part = torch.sigmoid(torch.gather(padded_scores, dim=1, index=left_t))
        right_part = 1.0 - torch.sigmoid(torch.gather(padded_scores, dim=1, index=right_t))
        left_p = torch.prod(left_part, dim=1)
        right_p = torch.prod(right_part, dim=1)
        probs = torch.mul(left_p, right_p)
        return torch.sum(-probs.log()) / batch_size
