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
python main.py --log_interval 200 --lr 0.1 --nhid 150 --nlayer 1 --epochs 5 --dropout 0 --model GRU --bptt 10 --batch_size 128
```


Training performance:

```
| epoch   1 |   200/  726 batches | lr 0.10 | ms/batch 134.03 | loss  6.07 | perplexity   433.54
| epoch   1 |   400/  726 batches | lr 0.10 | ms/batch 119.43 | loss  5.13 | perplexity   168.94
| epoch   1 |   600/  726 batches | lr 0.10 | ms/batch 114.21 | loss  4.96 | perplexity   141.98
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 90.46s | valid loss  4.97 | valid perplexity   144.64
-----------------------------------------------------------------------------------------
| epoch   2 |   200/  726 batches | lr 0.10 | ms/batch 110.90 | loss  4.78 | perplexity   119.56
| epoch   2 |   400/  726 batches | lr 0.10 | ms/batch 110.19 | loss  4.63 | perplexity   102.30
| epoch   2 |   600/  726 batches | lr 0.10 | ms/batch 112.23 | loss  4.57 | perplexity    96.82
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 84.09s | valid loss  4.87 | valid perplexity   129.88
-----------------------------------------------------------------------------------------
| epoch   3 |   200/  726 batches | lr 0.10 | ms/batch 112.57 | loss  4.53 | perplexity    92.92
| epoch   3 |   400/  726 batches | lr 0.10 | ms/batch 112.32 | loss  4.42 | perplexity    83.08
| epoch   3 |   600/  726 batches | lr 0.10 | ms/batch 109.83 | loss  4.39 | perplexity    80.52
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 83.53s | valid loss  4.84 | valid perplexity   126.24
-----------------------------------------------------------------------------------------
| epoch   4 |   200/  726 batches | lr 0.10 | ms/batch 112.65 | loss  4.39 | perplexity    80.44
| epoch   4 |   400/  726 batches | lr 0.10 | ms/batch 112.40 | loss  4.30 | perplexity    73.60
| epoch   4 |   600/  726 batches | lr 0.10 | ms/batch 110.25 | loss  4.28 | perplexity    72.08
-----------------------------------------------------------------------------------------
| end of epoch   4 | time: 83.47s | valid loss  4.83 | valid perplexity   125.47
-----------------------------------------------------------------------------------------
| epoch   5 |   200/  726 batches | lr 0.10 | ms/batch 111.02 | loss  4.29 | perplexity    73.20
| epoch   5 |   400/  726 batches | lr 0.10 | ms/batch 110.24 | loss  4.22 | perplexity    67.77
| epoch   5 |   600/  726 batches | lr 0.10 | ms/batch 110.31 | loss  4.20 | perplexity    66.73
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 82.85s | valid loss  4.83 | valid perplexity   125.83
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  4.80 | test perplexity   121.08
=========================================================================================
```


To generate a sample essay:
```bash
python generate.py
```

Generated sample essay:

```
policy joined was adjusted which electronics would trimmed would soil yesterday youth are cell scene quarter bought tv bidding air
east its new august over unit an rates yesterday stop quarter an least was will new surprisingly poverty would an
position gained government march recreational had fbi barrett its meeting unlike ghosts yesterday its la was an payout believe certificates
sector had its an ronald was going inc. one craft was statement proposed positive sold companies seen control sold its
default time region purchased automobiles completed was fm times are better executives times are m. its times are we times
are new paid was an its ltd. which increased texas was set would an relatively jr. quarter an cap shareholders
an every we its was both soviet them start scrutiny start-up sold its navy are are vice city yields users
had we investments building going jones dance balloting sold climate are we times are get was building would an gross
making stock exchange one over friday are better set would an often leaders especially keating was its short-term department default
cancer office further stock-market now calif. its was among real-estate similarity electric profitable prompting its drilling was ozone stars dividend
quarter are are had are are would fort yesterday traded de new are are afternoon caught traded was silver collapse
planning friday an paid attempt had stations electric friday rainbow yesterday bad sold are are was tone implicit are new
reaches sold tentatively academy both an quarters unit where exchange we times are better into off bring would its disk
its leaving quarter jazz exhibition lower ranked new profit-taking its seeking recall was inc. expense food we military denominations pacific
we an its federal both new loan traded plans off mediator sold would-be quarter are get quarter competitive would an
due cap its way we event small surrounded intensify forward was page-one off definitively both say yesterday waste we event
easier both its its an threat its would an pilots was an editorial-page r. unit often iran no fiscal where
difficult an increased face led its was ual off minutes recently we an position insurer rise had such building would
tv friday an evidence was an rates anything into air fundamentals branches its because could quarter an poland disaster proposed
will increased treasury during he sales defenders yesterday subordinated quotron was barnett sold having into prison contract stations chief companies
carriers disclosed would american october was wanting yesterday its implications charge was times are better would an texas would new
times are get drugs rates its stake filed now inc. one began had lucrative acquisition an companies seen buy-out filed
an toward ordinary would taking airways would cynthia business an texas would publishing its into areas executive east software an
overseas would an dreyfus spokeswoman yesterday cited respectable was this its an its would value quarter france subject every exchange
its impact poised recommendations howard business its had an anything negotiators new companies published was ual 1980s start eggs quick
anniversary will new politician bushel overcome parents provided chief electric fell toward published range economists was an campbell him an
its one than rights this linear an sells sold prison an run would an advanced wonder libel coaches was connections
potentially we yields de friday retailers business teaches building benchmark going quarter tumultuous adjusted this hot-dipped historic ana delivered unit
francs we p&g him an corporate was its collective now inc. condition expanded new face argue would american property lot
flows operations need unit an bankruptcy-law harold ice was contract age issues executive american personal coal was going net switched
our this its believe reported buck did ford written its into vice i lynch comprehensive department round unit tuesday give
visited real value had plants sony now must an winning canada act was quarter are fear now inc. which wires
both new prof. would dominated more sold think later pilots making an press would companies convince joint crimes second we
an individual had left deferring charlie quarter an collapse august redemptions vetoed we traders enough personal had holdings into an
options long opposed share ual salinger dick quarter say never age issues stands we center an degrees i front new
then provisions was this fundamental i year a. single-a new perlman business into an boost consider door board more subsidies
begin had announcement departments we treasury uses we stock-index very lawmakers engineering had which issues securities we jumped rejected has
call new underwriter was an currently businesses an negotiating would an beyond personal which issues neighborhoods investment leo did savings
parties both an recorded was board would an end would an session off tuesday laboratories had compiled display into mortgage
restructured would legislation them dropped meet we applying start going new stream would spending hard reported valued earthquake line significantly
would beginning one likely random was this fusion which known we escape did military both produce wells an covered its
was minimum would an chevron strategy we heavy an scores visitors we resort never new marshall its had inc. one
report respectively fees both number would an march more was report plans an room aimed main this its i east
southam business failing its quarter character him an losers was an solidarity must new climbed we attended times are better
quarter assume times are better plans bid executive exploring running because dead reductions chief an drexel would wood surprise yesterday
contracting was estimate start them discontinued third hard reported thus icahn deals would weighed are are seeking deep three an
built into an survival yesterday electrical we european large range an history gained proper we hollywood was computers en knock
which are companies are everybody unit an explained buying an history gained plans venture we ask takeover proposed could review
its had assume two was quarter germany are an source turning hard executive increased we suffered its business its currently
health had great estimated them countries had nation quarter entertainment pilots owns merely boosted franchise was players its an asian
```


## Reference

[1] [Classes for Fast Maximum Entropy Training](https://arxiv.org/abs/cs/0108006)

## Notes

The implementation of this language model is slightly different to my [TTIC_Deep_Learning_2018_Pareto_Competition](https://github.com/leimao/TTIC_Deep_Learning_2018_Pareto_Competition) which also uses the exact same two-layer hierarchical softmax. The word embedding layer and the two-layer hierarchical softmax layer were separated from the RNN model. But the parameters of them were still trained simutaneously. I would expect that the performance of this model and the one used for TTIC Deep Learning 2018 Pareto Competition the same. From the validation perplexity, it does seem so. However, when comparing the generated essays from the two models, I found significantly difference. I have not figured out the reason for that.
