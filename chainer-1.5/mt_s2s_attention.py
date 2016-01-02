# Switch to toggle CPU/GPU operation
USE_GPU = True

import sys
from argparse import ArgumentParser
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import util.generators as gens
from util.functions import trace, fill_batch
from util.vocabulary import Vocabulary

if USE_GPU:
  import cupy as np
else:
  import numpy as np

def parse_args():
  def_vocab = 1000
  def_embed = 100
  def_hidden = 200
  def_epoch = 10
  def_minibatch = 64
  def_generation_limit = 128

  p = ArgumentParser(description='Attentional neural machine trainslation')

  p.add_argument('mode', help='\'train\' or \'test\'')
  p.add_argument('source', help='[in] source corpus')
  p.add_argument('target', help='[in/out] target corpus')
  p.add_argument('model', help='[in/out] model file')
  p.add_argument('--vocab', default=def_vocab, metavar='INT', type=int,
    help='vocabulary size (default: %d)' % def_vocab)
  p.add_argument('--embed', default=def_embed, metavar='INT', type=int,
    help='embedding layer size (default: %d)' % def_embed)
  p.add_argument('--hidden', default=def_hidden, metavar='INT', type=int,
    help='hidden layer size (default: %d)' % def_hidden)
  p.add_argument('--epoch', default=def_epoch, metavar='INT', type=int,
    help='number of training epoch (default: %d)' % def_epoch)
  p.add_argument('--minibatch', default=def_minibatch, metavar='INT', type=int,
    help='minibatch size (default: %d)' % def_minibatch)
  p.add_argument('--generation-limit', default=def_generation_limit, metavar='INT', type=int,
    help='maximum number of words to be generated for test input')

  args = p.parse_args()

  # check args
  try:
    if args.mode not in ['train', 'test']: raise ValueError('you must set mode = \'train\' or \'test\'')
    if args.vocab < 1: raise ValueError('you must set --vocab >= 1')
    if args.embed < 1: raise ValueError('you must set --embed >= 1')
    if args.hidden < 1: raise ValueError('you must set --hidden >= 1')
    if args.epoch < 1: raise ValueError('you must set --epoch >= 1')
    if args.minibatch < 1: raise ValueError('you must set --minibatch >= 1')
    if args.generation_limit < 1: raise ValueError('you must set --generation-limit >= 1')
  except Exception as ex:
    p.print_usage(file=sys.stderr)
    print(ex, file=sys.stderr)
    sys.exit()

  return args

def my_zeros(shape, dtype):
  return Variable(np.zeros(shape, dtype=dtype))

def my_array(array, dtype):
  return Variable(np.array(array, dtype=dtype))

class SrcEmbed(Chain):
  def __init__(self, vocab_size, embed_size):
    super(SrcEmbed, self).__init__(
        xe = links.EmbedID(vocab_size, embed_size),
    )

  def __call__(self, x):
    return functions.tanh(self.xe(x))

class Encoder(Chain):
  def __init__(self, embed_size, hidden_size):
    super(Encoder, self).__init__(
        xh = links.Linear(embed_size, 4 * hidden_size),
        hh = links.Linear(hidden_size, 4 * hidden_size),
    )

  def __call__(self, x, c, h):
    return functions.lstm(c, self.xh(x) + self.hh(h))

class Attention(Chain):
  def __init__(self, hidden_size):
    super(Attention, self).__init__(
        aw = links.Linear(hidden_size, hidden_size),
        bw = links.Linear(hidden_size, hidden_size),
        pw = links.Linear(hidden_size, hidden_size),
        we = links.Linear(hidden_size, 1),
    )
    self.hidden_size = hidden_size

  def __call__(self, a_list, b_list, p):
    batch_size = p.data.shape[0]
    e_list = []
    sum_e = my_zeros((batch_size, 1), np.float32)
    for a, b in zip(a_list, b_list):
      w = functions.tanh(self.aw(a) + self.bw(b) + self.pw(p))
      e = functions.exp(self.we(w))
      e_list.append(e)
      sum_e += e
    ZEROS = my_zeros((batch_size, self.hidden_size), np.float32)
    aa = ZEROS
    bb = ZEROS
    for a, b, e in zip(a_list, b_list, e_list):
      e /= sum_e
      aa += functions.reshape(functions.batch_matmul(a, e), (batch_size, self.hidden_size))
      bb += functions.reshape(functions.batch_matmul(b, e), (batch_size, self.hidden_size))
    return aa, bb

class Decoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Decoder, self).__init__(
        ye = links.EmbedID(vocab_size, embed_size),
        eh = links.Linear(embed_size, 4 * hidden_size),
        hh = links.Linear(hidden_size, 4 * hidden_size),
        ah = links.Linear(hidden_size, 4 * hidden_size),
        bh = links.Linear(hidden_size, 4 * hidden_size),
        hf = links.Linear(hidden_size, embed_size),
        fy = links.Linear(embed_size, vocab_size),
    )

  def __call__(self, y, c, h, a, b):
    e = functions.tanh(self.ye(y))
    c, h = functions.lstm(c, self.eh(e) + self.hh(h) + self.ah(a) + self.bh(b))
    f = functions.tanh(self.hf(h))
    return self.fy(f), c, h

class AttentionMT(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(AttentionMT, self).__init__(
        emb = SrcEmbed(vocab_size, embed_size),
        fenc = Encoder(embed_size, hidden_size),
        benc = Encoder(embed_size, hidden_size),
        att = Attention(hidden_size),
        dec = Decoder(vocab_size, embed_size, hidden_size),
    )
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size

  def reset(self, batch_size):
    self.zerograds()
    self.x_list = []

  def embed(self, x):
    self.x_list.append(self.emb(x))

  def encode(self):
    src_len = len(self.x_list)
    batch_size = self.x_list[0].data.shape[0]
    ZEROS = my_zeros((batch_size, self.hidden_size), np.float32)
    c = ZEROS
    a = ZEROS
    a_list = []
    for x in self.x_list:
      c, a = self.fenc(x, c, a)
      a_list.append(a)
    c = ZEROS
    b = ZEROS
    b_list = []
    for x in reversed(self.x_list):
      c, b = self.benc(x, c, b)
      b_list.insert(0, b)
    self.a_list = a_list
    self.b_list = b_list
    self.c = ZEROS
    self.h = ZEROS

  def decode(self, y):
    aa, bb = self.att(self.a_list, self.b_list, self.h)
    y, self.c, self.h = self.dec(y, self.c, self.h, aa, bb)
    return y

  def save_spec(self, filename):
    with open(filename, 'w') as fp:
      print(self.vocab_size, file=fp)
      print(self.embed_size, file=fp)
      print(self.hidden_size, file=fp)

  @staticmethod
  def load_spec(filename):
    with open(filename) as fp:
      vocab_size = int(next(fp))
      embed_size = int(next(fp))
      hidden_size = int(next(fp))
      return AttentionMT(vocab_size, embed_size, hidden_size)

def forward(src_batch, trg_batch, src_vocab, trg_vocab, attmt, is_training, generation_limit):
  batch_size = len(src_batch)
  src_len = len(src_batch[0])
  trg_len = len(trg_batch[0]) if trg_batch else 0
  src_stoi = src_vocab.stoi
  trg_stoi = trg_vocab.stoi
  trg_itos = trg_vocab.itos
  attmt.reset(batch_size)

  x = my_array([src_stoi('<s>') for _ in range(batch_size)], np.int32)
  attmt.embed(x)
  for l in range(src_len):
    x = my_array([src_stoi(src_batch[k][l]) for k in range(batch_size)], np.int32)
    attmt.embed(x)
  x = my_array([src_stoi('</s>') for _ in range(batch_size)], np.int32)
  attmt.embed(x)

  attmt.encode()
  
  t = my_array([trg_stoi('<s>') for _ in range(batch_size)], np.int32)
  hyp_batch = [[] for _ in range(batch_size)]

  if is_training:
    loss = my_zeros((), np.float32)
    for l in range(trg_len):
      y = attmt.decode(t)
      t = my_array([trg_stoi(trg_batch[k][l]) for k in range(batch_size)], np.int32)
      loss += functions.softmax_cross_entropy(y, t)
      output = cuda.to_cpu(y.data.argmax(1))
      for k in range(batch_size):
        hyp_batch[k].append(trg_itos(output[k]))
    return hyp_batch, loss
  
  else:
    while len(hyp_batch[0]) < generation_limit:
      y = attmt.decode(t)
      output = cuda.to_cpu(y.data.argmax(1))
      t = my_array(output, np.int32)
      for k in range(batch_size):
        hyp_batch[k].append(trg_itos(output[k]))
      if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)):
        break

    return hyp_batch

def train(args):
  trace('making vocaburaries ...')
  src_vocab = Vocabulary.new(gens.word_list(args.source), args.vocab)
  trg_vocab = Vocabulary.new(gens.word_list(args.target), args.vocab)

  trace('making model ...')
  attmt = AttentionMT(args.vocab, args.embed, args.hidden)
  if USE_GPU:
    attmt.to_gpu()

  for epoch in range(args.epoch):
    trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
    trained = 0
    gen1 = gens.word_list(args.source)
    gen2 = gens.word_list(args.target)
    gen3 = gens.batch(gens.sorted_parallel(gen1, gen2, 100 * args.minibatch), args.minibatch)
    opt = optimizers.AdaGrad(lr = 0.01)
    opt.setup(attmt)
    opt.add_hook(optimizer.GradientClipping(5))

    for src_batch, trg_batch in gen3:
      src_batch = fill_batch(src_batch)
      trg_batch = fill_batch(trg_batch)
      K = len(src_batch)
      hyp_batch, loss = forward(src_batch, trg_batch, src_vocab, trg_vocab, attmt, True, 0)
      loss.backward()
      opt.update()

      for k in range(K):
        trace('epoch %3d/%3d, sample %8d' % (epoch + 1, args.epoch, trained + k + 1))
        trace('  src = ' + ' '.join([x if x != '</s>' else '*' for x in src_batch[k]]))
        trace('  trg = ' + ' '.join([x if x != '</s>' else '*' for x in trg_batch[k]]))
        trace('  hyp = ' + ' '.join([x if x != '</s>' else '*' for x in hyp_batch[k]]))

      trained += K

    trace('saving model ...')
    prefix = args.model + '.%03.d' % (epoch + 1)
    src_vocab.save(prefix + '.srcvocab')
    trg_vocab.save(prefix + '.trgvocab')
    attmt.save_spec(prefix + '.spec')
    serializers.save_hdf5(prefix + '.weights', attmt)

  trace('finished.')

def test(args):
  trace('loading model ...')
  src_vocab = Vocabulary.load(args.model + '.srcvocab')
  trg_vocab = Vocabulary.load(args.model + '.trgvocab')
  attmt = AttentionMT.load_spec(args.model + '.spec')
  if USE_GPU:
    attmt.to_gpu()
  serializers.load_hdf5(args.model + '.weights', attmt)
  
  trace('generating translation ...')
  generated = 0

  with open(args.target, 'w') as fp:
    for src_batch in gens.batch(gens.word_list(args.source), args.minibatch):
      src_batch = fill_batch(src_batch)
      K = len(src_batch)

      trace('sample %8d - %8d ...' % (generated + 1, generated + K))
      hyp_batch = forward(src_batch, None, src_vocab, trg_vocab, attmt, False, args.generation_limit)

      for hyp in hyp_batch:
        hyp.append('</s>')
        hyp = hyp[:hyp.index('</s>')]
        print(' '.join(hyp), file=fp)

      generated += K

  trace('finished.')

def main():
  args = parse_args()
  if args.mode == 'train': train(args)
  elif args.mode == 'test': test(args)

if __name__ == '__main__':
  main()

