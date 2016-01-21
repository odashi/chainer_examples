import sys
import numpy
from argparse import ArgumentParser
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import util.generators as gens
from util.functions import trace, fill_batch
from util.vocabulary import Vocabulary

def parse_args():
  def_gpu_device = 0
  def_vocab = 1000
  def_embed = 100
  def_hidden = 200
  def_epoch = 10
  def_minibatch = 64
  def_generation_limit = 128

  p = ArgumentParser(
    description='Encoder-decoder neural machine trainslation',
    usage=
      '\n  %(prog)s train [options] source model'
      '\n  %(prog)s test source model'
      '\n  %(prog)s -h',
  )

  p.add_argument('mode', help='\'train\' or \'test\'')
  p.add_argument('source', help='[in] source corpus')
  p.add_argument('target', help='[in/out] target corpus')
  p.add_argument('model', help='[in/out] model file')
  p.add_argument('--use-gpu', action='store_true', default=False,
    help='use GPU calculation')
  p.add_argument('--gpu-device', default=def_gpu_device, metavar='INT', type=int,
    help='GPU device ID to be used (default: %(default)d)')
  p.add_argument('--vocab', default=def_vocab, metavar='INT', type=int,
    help='vocabulary size (default: %(default)d)')
  p.add_argument('--embed', default=def_embed, metavar='INT', type=int,
    help='embedding layer size (default: %(default)d)')
  p.add_argument('--hidden', default=def_hidden, metavar='INT', type=int,
    help='hidden layer size (default: %(default)d)')
  p.add_argument('--epoch', default=def_epoch, metavar='INT', type=int,
    help='number of training epoch (default: %(default)d)')
  p.add_argument('--minibatch', default=def_minibatch, metavar='INT', type=int,
    help='minibatch size (default: %(default)d)')
  p.add_argument('--generation-limit', default=def_generation_limit, metavar='INT', type=int,
    help='maximum number of words to be generated for test input (default: %(default)d)')

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

class XP:
  __lib = None

  @staticmethod
  def set_library(args):
    if args.use_gpu:
      XP.__lib = cuda.cupy
      cuda.get_device(args.gpu_device).use()
    else:
      XP.__lib = numpy

  @staticmethod
  def __zeros(shape, dtype):
    return Variable(XP.__lib.zeros(shape, dtype=dtype))

  @staticmethod
  def fzeros(shape):
    return XP.__zeros(shape, XP.__lib.float32)

  @staticmethod
  def __array(array, dtype):
    return Variable(XP.__lib.array(array, dtype=dtype))

  @staticmethod
  def iarray(array):
    return XP.__array(array, XP.__lib.int32)

  @staticmethod
  def farray(array):
    return XP.__array(array, XP.__lib.float32)

class Encoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Encoder, self).__init__(
        xe = links.EmbedID(vocab_size, embed_size),
        eh = links.Linear(embed_size, 4 * hidden_size),
        hh = links.Linear(hidden_size, 4 * hidden_size),
    )

  def __call__(self, x, c, h):
    e = functions.tanh(self.xe(x))
    return functions.lstm(c, self.eh(e) + self.hh(h)) 

class Decoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(Decoder, self).__init__(
        ye = links.EmbedID(vocab_size, embed_size),
        eh = links.Linear(embed_size, 4 * hidden_size),
        hh = links.Linear(hidden_size, 4 * hidden_size),
        hf = links.Linear(hidden_size, embed_size),
        fy = links.Linear(embed_size, vocab_size),
    )

  def __call__(self, y, c, h):
    e = functions.tanh(self.ye(y))
    c, h = functions.lstm(c, self.eh(e) + self.hh(h))
    f = functions.tanh(self.hf(h))
    return self.fy(f), c, h

class EncoderDecoder(Chain):
  def __init__(self, vocab_size, embed_size, hidden_size):
    super(EncoderDecoder, self).__init__(
        enc = Encoder(vocab_size, embed_size, hidden_size),
        dec = Decoder(vocab_size, embed_size, hidden_size),
    )
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size

  def reset(self, batch_size):
    self.zerograds()
    self.c = XP.fzeros((batch_size, self.hidden_size))
    self.h = XP.fzeros((batch_size, self.hidden_size))

  def encode(self, x):
    self.c, self.h = self.enc(x, self.c, self.h)

  def decode(self, y):
    y, self.c, self.h = self.dec(y, self.c, self.h)
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
      return EncoderDecoder(vocab_size, embed_size, hidden_size)

def forward(src_batch, trg_batch, src_vocab, trg_vocab, encdec, is_training, generation_limit):
  batch_size = len(src_batch)
  src_len = len(src_batch[0])
  trg_len = len(trg_batch[0]) if trg_batch else 0
  src_stoi = src_vocab.stoi
  trg_stoi = trg_vocab.stoi
  trg_itos = trg_vocab.itos
  encdec.reset(batch_size)

  x = XP.iarray([src_stoi('</s>') for _ in range(batch_size)])
  encdec.encode(x)
  for l in reversed(range(src_len)):
    x = XP.iarray([src_stoi(src_batch[k][l]) for k in range(batch_size)])
    encdec.encode(x)
  
  t = XP.iarray([trg_stoi('<s>') for _ in range(batch_size)])
  hyp_batch = [[] for _ in range(batch_size)]

  if is_training:
    loss = XP.fzeros(())
    for l in range(trg_len):
      y = encdec.decode(t)
      t = XP.iarray([trg_stoi(trg_batch[k][l]) for k in range(batch_size)])
      loss += functions.softmax_cross_entropy(y, t)
      output = cuda.to_cpu(y.data.argmax(1))
      for k in range(batch_size):
        hyp_batch[k].append(trg_itos(output[k]))
    return hyp_batch, loss
  
  else:
    while len(hyp_batch[0]) < generation_limit:
      y = encdec.decode(t)
      output = cuda.to_cpu(y.data.argmax(1))
      t = XP.iarray(output)
      for k in range(batch_size):
        hyp_batch[k].append(trg_itos(output[k]))
      if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)):
        break

    return hyp_batch

def train(args):
  trace('making vocabularies ...')
  src_vocab = Vocabulary.new(gens.word_list(args.source), args.vocab)
  trg_vocab = Vocabulary.new(gens.word_list(args.target), args.vocab)

  trace('making model ...')
  encdec = EncoderDecoder(args.vocab, args.embed, args.hidden)
  if args.use_gpu:
    encdec.to_gpu()

  for epoch in range(args.epoch):
    trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
    trained = 0
    gen1 = gens.word_list(args.source)
    gen2 = gens.word_list(args.target)
    gen3 = gens.batch(gens.sorted_parallel(gen1, gen2, 100 * args.minibatch), args.minibatch)
    opt = optimizers.AdaGrad(lr = 0.01)
    opt.setup(encdec)
    opt.add_hook(optimizer.GradientClipping(5))

    for src_batch, trg_batch in gen3:
      src_batch = fill_batch(src_batch)
      trg_batch = fill_batch(trg_batch)
      K = len(src_batch)
      hyp_batch, loss = forward(src_batch, trg_batch, src_vocab, trg_vocab, encdec, True, 0)
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
    encdec.save_spec(prefix + '.spec')
    serializers.save_hdf5(prefix + '.weights', encdec)

  trace('finished.')

def test(args):
  trace('loading model ...')
  src_vocab = Vocabulary.load(args.model + '.srcvocab')
  trg_vocab = Vocabulary.load(args.model + '.trgvocab')
  encdec = EncoderDecoder.load_spec(args.model + '.spec')
  if args.use_gpu:
    encdec.to_gpu()
  serializers.load_hdf5(args.model + '.weights', encdec)
  
  trace('generating translation ...')
  generated = 0

  with open(args.target, 'w') as fp:
    for src_batch in gens.batch(gens.word_list(args.source), args.minibatch):
      src_batch = fill_batch(src_batch)
      K = len(src_batch)

      trace('sample %8d - %8d ...' % (generated + 1, generated + K))
      hyp_batch = forward(src_batch, None, src_vocab, trg_vocab, encdec, False, args.generation_limit)

      for hyp in hyp_batch:
        hyp.append('</s>')
        hyp = hyp[:hyp.index('</s>')]
        print(' '.join(hyp), file=fp)

      generated += K

  trace('finished.')

def main():
  args = parse_args()
  XP.set_library(args)
  if args.mode == 'train': train(args)
  elif args.mode == 'test': test(args)

if __name__ == '__main__':
  main()

