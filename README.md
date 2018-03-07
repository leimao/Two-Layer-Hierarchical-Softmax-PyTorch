# Two_Layer_Hierarchical_Softmax_PyTorch

Lei Mao

University of Chicago

## Introduction

Hierarchical softmax is a softmax alternative to the full softmax used in language modeling when the corpus is large. The simplest hierarhical softmax is the [two-layer hierarchical softmax](https://arxiv.org/abs/cs/0108006). Theano has a version of [two-layer hierarchical softmax](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.h_softmax) which could be easily employed by the users. In contrast, Facebook PyTorch does not provide any softmax alternatives at all. Based on his code, I implemented the two-layer hierarchical softmax using PyTorch.


## Dependencies

Python 3.5

PyTorch 0.3

Numpy

## Files

The two-layer hierarchical softmax, and other helper classes and functions are all coded in ``utils.py`` file. I also modifed Facebook [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model) for the implementation test.

```
.
├── data
│   └── ptb
│       ├── test.txt
│       ├── train.txt
│       └── valid.txt
├── data.py
├── generate.py
├── LICENSE.md
├── main.py
├── model.py
├── README.md
└── utils.py
```


## Usage

I added several options to the Word-level language modeling RNN api. But some of the options, such as ``tied``, might not be valid any more due to the modification in the code. I will fix these in the future if I get time.

```
optional arguments:
  -h, --help             show this help message and exit
  --data DATA            location of the data corpus
  --model MODEL          type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE        size of word embeddings
  --nhid NHID            number of hidden units per layer
  --nlayers NLAYERS      number of layers
  --lr LR                initial learning rate
  --clip CLIP            gradient clipping
  --epochs EPOCHS        upper epoch limit
  --batch-size N         batch size
  --bptt BPTT            sequence length
  --dropout DROPOUT      dropout applied to layers (0 = no dropout)
  --decay DECAY          learning rate decay per epoch
  --tied                 tie the word embedding and softmax weights
  --seed SEED            random seed
  --log-interval N       report interval
  --save SAVE            path to save the final model
```


To train a language model using sampled softmax:

```bash
python main.py --log_interval 200 --lr 0.1 --nhid 150 --nlayer 1 --epochs 5 --dropout 0 --model GRU --bptt 12 --batch_size 128 --seed 110
```


The training result on single-core CPU:

```
| epoch   1 |   200/  605 batches | lr 0.10 | ms/batch 228.70 | loss  6.05 | perplexity   423.82
| epoch   1 |   400/  605 batches | lr 0.10 | ms/batch 200.08 | loss  5.08 | perplexity   160.35
| epoch   1 |   600/  605 batches | lr 0.10 | ms/batch 195.75 | loss  4.90 | perplexity   134.31
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 129.48s | valid loss  4.97 | valid perplexity   143.79
-----------------------------------------------------------------------------------------
| epoch   2 |   200/  605 batches | lr 0.10 | ms/batch 190.06 | loss  4.77 | perplexity   117.85
| epoch   2 |   400/  605 batches | lr 0.10 | ms/batch 190.08 | loss  4.61 | perplexity   100.86
| epoch   2 |   600/  605 batches | lr 0.10 | ms/batch 188.95 | loss  4.55 | perplexity    94.38
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 118.26s | valid loss  4.87 | valid perplexity   129.87
-----------------------------------------------------------------------------------------
| epoch   3 |   200/  605 batches | lr 0.10 | ms/batch 189.51 | loss  4.52 | perplexity    92.00
| epoch   3 |   400/  605 batches | lr 0.10 | ms/batch 188.86 | loss  4.42 | perplexity    82.74
| epoch   3 |   600/  605 batches | lr 0.10 | ms/batch 189.52 | loss  4.37 | perplexity    78.85
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 118.05s | valid loss  4.84 | valid perplexity   125.87
-----------------------------------------------------------------------------------------
| epoch   4 |   200/  605 batches | lr 0.10 | ms/batch 190.62 | loss  4.37 | perplexity    79.10
| epoch   4 |   400/  605 batches | lr 0.10 | ms/batch 189.51 | loss  4.29 | perplexity    72.82
| epoch   4 |   600/  605 batches | lr 0.10 | ms/batch 189.46 | loss  4.25 | perplexity    70.25
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 118.36s | valid loss  4.83 | valid perplexity   125.37
-----------------------------------------------------------------------------------------
| epoch   5 |   200/  605 batches | lr 0.10 | ms/batch 190.91 | loss  4.27 | perplexity    71.59
| epoch   5 |   400/  605 batches | lr 0.10 | ms/batch 188.86 | loss  4.20 | perplexity    66.79
| epoch   5 |   600/  605 batches | lr 0.10 | ms/batch 189.74 | loss  4.17 | perplexity    64.81
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 118.34s | valid loss  4.84 | valid perplexity   126.11
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  4.79 | test perplexity   120.41
=========================================================================================
```

By the way, to restrict the usage of CPU to single core for PyTorch in Linux environment, please add the following into the ``~/.bashrc``:

```
# OpenMP
# Restrict the number of threads used in OpenMP to 1
export OMP_NUM_THREADS=1
```

Please also remember to refresh the ``~/.bashrc`` by:

```bash
source ~/.bashrc
```


To generate a sample essay:
```bash
python generate.py
```

```
up came N million marks but that includes $ N to $ N <eos> the mega-issues about the most recent
small deterioration in which mr. <unk> 's tabloid as well among continued direction <eos> the business has N N of
the year earlier while just <unk> plaintiffs nations had been falling into N roads and african suspension of a imports
<eos> in the N ciba-geigy multiple cells sent to a <unk> japanese workers have south african new haven and hampshire
restaurant well <eos> good for a market for manufacturers helps whenever no longer <unk> or when imports are <unk> in
the economy <eos> so some strong experience with a unified discovered that no recovery will be operating out plans <eos>
the fed then in effect yesterday makes a ship within six years of which now <unk> soviet computing and unions
cleared out of philip marcos 's production <eos> the labor department introduced that the problems of balls has a long
off of N buildings of <unk> cars <eos> the eye seems <unk> <eos> night after all chromosome was stopped almost
selling overtime per commuters toward cold data <eos> it plans last year in three of the safety syndrome that cars
ibm coast codes had access to the equipment to appeal that new patent stores is <unk> with a <unk> price
<eos> it was viewed the situation says adding it caused a room mostly line with low sales inflation <eos> but
not the u.s. <eos> people now have <unk> these days than just as they refer <eos> in the proportion of
carry viewers sharply furriers like researchers at mixed in computers <eos> sales of white makers are in a place at
their u.s. international association ltd <eos> ford once the government was discovered that university 's sales recently designed also to
learn as a labor department permanent debate aroused on a sale of N a year <eos> among other tasks by
women 's decision hugo recently raised both the administration <eos> it may follow washington N years at the university of
city city <eos> however if both sides are highly <unk> and kent <unk> filters for behavior in the <unk> <eos>
the museum of <unk> advised the net gap during a single postal home in the used joint refinery with genentech
inc. and by federal canadian standards and truck service council and chief industry 's operating economic data development company recently
has n't been <unk> in the future remains sluggish in japan society to only its metropolitan european america nations <eos>
the country once reached for we were especially <unk> in the facility in technology <eos> delta france experienced over the
latest <unk> in san terms to area bowed N in the san francisco markets in the west german oil prices
via staging workstations a <unk> <unk> <eos> <unk> county there is seeking more to be identified engines <eos> <unk> corp.
the <unk> <unk> for possible <unk> n.y. named mikhail lawson and new york-based although the project could n't be a
<unk> over <eos> the boss was president and chief executive officer says he 'd quickly since <unk> N years ago
they do n't believe that the plan did n't identify comment who tend to remove him the turn of first
optical address <eos> but instead of the filing is n't syndicated by their departures on a profit by the title
the approximately payments in fiscal N <eos> some lawyers must reject the current incinerator that might seek to be even
color <eos> mr. indexing attorney n't passed the leveraged tool so he explained the fact that real estate in various
<unk> he says the paper and or losses might could required a shift either to obtain greater than he is
<unk> from <unk> the <unk> <eos> he admits <unk> <unk> to become president <unk> the <unk> of the cancer fuji
to get into a safe marina banking company that mr. tharp said mr. corry were willing to have made an
agreement with individual vessels in richmond instead the assets <eos> but neither require mr. trump 's separate meetings had been
part of the current debate over a filing in mr. white 's source notorious constitutional gulf power <eos> said judge
reckless <unk> that let mr. lorin the <unk> withdraw and may have been the bork 's aborted risks saying representatives
about limited economic hopes <unk> with xerox properties their $ N million bid <eos> among other other paper issues related
to the restructuring includes a huge code couple of west application by last year announced last year and by a
sugar industry officials in the july N <eos> today market vowed to <unk> market makers in european european markets he
began sitting in N chemical volume last year <eos> there will be one significant practice of inflation he 'll help
keep itself on <unk> <eos> the part of the futures change will produce a potential standard u.s. and focus on
research for international businesses under <unk> japan <eos> joseph refer corp. an estimated N N interest rates rose to N
pence $ N an share <eos> <unk> bonds were delayed to $ N in over-the-counter trading at N common shares
outstanding <eos> commercial of new machines with imposes to short-term u.s. financial abuse after marginal dragging russell james <eos> the
<unk> <unk> which is regarded as inception and in sacramento <eos> i had ever linked to her job <eos> the
congressional behavior of the major government that will be released today by <unk> & looms cases to let teddy <unk>
that case challenges at any people think likewise those boards and there will be an issue requirement <eos> other <unk>
abuse from investors know what he makes them for that eugene rather than aspects of the unsuccessful run team were
based on civil alternatives and others should respond says reporters <eos> if the middle of the poverty cure are far
worse not <unk> a leads is to compaq 's system because the <unk> was <unk> and <unk> of the standardized
```


## Reference

[1] [Classes for Fast Maximum Entropy Training](https://arxiv.org/abs/cs/0108006)
