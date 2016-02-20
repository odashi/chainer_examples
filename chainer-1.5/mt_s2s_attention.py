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
    description='Attentional neural machine trainslation',
    usage=
      '\n  %(prog)s train [options] source target model'
      '\n  %(prog)s test source target model'
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
  def __nonzeros(shape, dtype, val):
    return Variable(val * XP.__lib.ones(shape, dtype=dtype))

  @staticmethod
  def fnonzeros(shape, val=1):
    return XP.__nonzeros(shape, XP.__lib.float32, val)

  @staticmethod
  def __array(array, dtype):
    return Variable(XP.__lib.array(array, dtype=dtype))

  @staticmethod
  def iarray(array):
    return XP.__array(array, XP.__lib.int32)

  @staticmethod
  def farray(array):
    return XP.__array(array, XP.__lib.float32)

class SrcEmbed(Chain):
  def __init__(self, vocab_size, embed_size):
    super(SrcEmbed, self).__init__(
        xe = links.EmbedID(vocab_size, embed_size),
    )

  def __call__(self, x):
    return functions.tanh(self.xe(x))




class MultiLayerStatefulLSTMEncoder(ChainList):
  """
  This is an implementation of a Multilayered Stateful LSTM.
  The underlying idea is to simply stack multiple LSTMs where the LSTM at the bottom takes the regular input,
  and the LSTMs after that simply take the outputs (represented by h) of the previous LSMTs as inputs.
  This is simply an analogous version of the Multilayered Stateless LSTM Encoder where the LSTM states are kept hidden.
  This LSTM is to be called only by passing the input (x).
  To access the cell states you must call the "get_states" function with parameter "num_layers" indicating the number of layers.
  Although the cell outputs for each layer are returned, typically only the one of the topmost layer is used for various purposes like attention.
  Note that in Tensorflow the concept of "number of attention heads" is used which probably points to attention using the output of each layer.

  Args: 
        embed_size - The size of embeddings of the inputs
        hidden_size - The size of the hidden layer representation of the RNN
        num_layers - The number of layers of the RNN (Indicates the number of RNNS stacked on top of each other)

  Attributes: 
        num_layers: Indicates the number of layers in the RNN
  User Defined Methods:
        get_states: This simply returns the latest cell states (c) as an array for all layers.

  """

  def __init__(self, embed_size, hidden_size, num_layers):
    super(MultiLayerStatefulLSTMEncoder, self).__init__()
    self.add_link(links.LSTM(embed_size,hidden_size))
    for i in range(1, num_layers):
      self.add_link(links.LSTM(hidden_size, hidden_size))
    self.num_layers = num_layers
      
  def __call__(self, x):
    """
    Updates the internal state and returns the RNN outputs for each layer as a list.

    Args:
        x : A new batch from the input sequence.

    Returns:
        A list of the outputs (h) of updated RNN units over all the layers.

    """
    h_list = []
    h_curr = self[0](x)
    h_list.append(h_curr)
    for i in range(1,self.num_layers):
      h_curr = self[1](h_curr)
      h_list.append(h_curr)
    return h_list

  def get_states():
    c_list = []
    for i in range(self.num_layers):
      c_list.append(self[i].c)
    return c_list

class MultiLayerStatelessLSTMEncoder(ChainList):
  """
  This is an implementation of a Multilayered Stateless LSTM.
  The underlying idea is to simply stack multiple LSTMs where the LSTM at the bottom takes the regular input,
  and the LSTMs after that simply take the outputs (represented by h) of the previous LSMTs as inputs.
  This is simply an analogous version of the Multilayered Stateful LSTM Encoder where the LSTM states are not hidden.
  You have to pass the previous cell states (c) and outputs (h) along with the input (x) when calling the LSTM.
  Although the cell outputs for each layer are returned, typically only the one of the topmost layer is used for various purposes like attention.
  Note that in Tensorflow the concept of "number of attention heads" is used which probably points to attention using the output of each layer.

  Args: 
        embed_size - The size of embeddings of the inputs
        hidden_size - The size of the hidden layer representation of the RNN
        num_layers - The number of layers of the RNN (Indicates the number of RNNS stacked on top of each other)

  Attributes: 
        num_layers: Indicates the number of layers in the RNN
  User Defined Methods:
        
  """
  def __init__(self, embed_size, hidden_size, num_layers):
    super(MultiLayerStatelessLSTMEncoder, self).__init__()

    self.add_link(links.Linear(embed_size, 4 * hidden_size))
    self.add_link(links.Linear(hidden_size, 4 * hidden_size))
    for i in range(1,num_layers):
      self.add_link(links.Linear(hidden_size, 4 * hidden_size))
      self.add_link(links.Linear(hidden_size, 4 * hidden_size))
    self.num_layers = num_layers
  def __call__(self, x, c, h):
    """
    Updates the internal state and returns the RNN outputs for each layer as a list.

    Args:
        x : A new batch from the input sequence.
        c : The list of the previous cell states.
        h : The list of the previous cell outputs.
    Returns:
        A list of the outputs (h) and another of the states (c) of the updated RNN units over all the layers.

    """
    c_list = []
    h_list = []
    c_curr, h_curr = functions.lstm(c[0], self[0](x) + self[1](h[0]))
    c_list.append(c_curr)
    h_list.append(h_curr)
    for i in range(1,self.num_layers):
      c_curr, h_curr = functions.lstm(c[i], self[(i*num_layers)+0](h_curr) + self[(i*num_layers)+1](h[i]))
      c_list.append(c_curr)
      h_list.append(h_curr)
    return c_list, h_list

class MultiLayerGRUEncoder(ChainList):
  """
  This is an implementation of a Multilayered Stateless GRU.
  The underlying idea is to simply stack multiple GRUs where the GRU at the bottom takes the regular input,
  and the GRUs after that simply take the outputs (represented by h) of the previous GRUs as inputs.
  You have to pass the previous cell outputs (h) along with the input (x) when calling the LSTM.
  The implementation for the Stateful GRU just saves the cell state and thus its multilayered version wont be implemented unless demanded.

  Args: 
        embed_size - The size of embeddings of the inputs
        hidden_size - The size of the hidden layer representation of the RNN
        num_layers - The number of layers of the RNN (Indicates the number of RNNS stacked on top of each other)

  Attributes: 
        num_layers: Indicates the number of layers in the RNN
  User Defined Methods:
        
  """

  def __init__(self, embed_size, hidden_size, num_layers):
    super(MultiLayerGRUEncoder, self).__init__()
    self.add_link(links.GRU(hidden_size,embed_size))
    for i in num_layers:
      self.add_link(links.GRU(hidden_size,hidden_size))
    self.num_layers = num_layers

  def __call__(self, x, h):
    """
    Updates the internal state and returns the RNN outputs for each layer as a list.

    Args:
        x : A new batch from the input sequence.
        h : The list of the previous cell outputs.
    Returns:
        A list of the outputs (h) of the updated RNN units over all the layers.

    """
    h_list = []
    h_curr = self[0](h[0], x)
    h_list.append(h_curr)
    for i in range(1,self.num_layers):
      h_curr = self[i](h[i], h_curr)
      h_list.append(h_curr)
    return h_list


class GRUEncoder(Chain):
  
  """
  This is just the same Encoder as below.
  The only difference is that the RNN cell is a GRU.
  

  Args: 
        embed_size - The size of embeddings of the inputs
        hidden_size - The size of the hidden layer representation of the RNN
        

  Attributes: 
        
  User Defined Methods:
  
  """

  def __init__(self, embed_size, hidden_size):
    super(Encoder, self).__init__(
        GRU = links.GRU(embed_size, hidden_size),
    )

  def __call__(self, x):
    """
    Updates the internal state and returns the RNN output (h).
    Note that for a GRU the internal state is the same as the output. (c and h are the same)

    Args:
        x : A new batch from the input sequence.

    Returns:
        The output (h) of updated RNN unit.

    """
    return self.GRU(x)

class StatefulEncoder(Chain):
  
  """
  This is just the same Encoder as below.
  The only difference is that the LSTM class implementation is used instead of the LSTM function.
  Instead of explicitly defining the LSTM components, the LSTM class encapsulates these components making the Encoder look simpler.

  Args: 
        embed_size - The size of embeddings of the inputs
        hidden_size - The size of the hidden layer representation of the RNN
        

  Attributes: 
        
  User Defined Methods:
        get_state: This simply returns the latest cell state (c).
  """

  def __init__(self, embed_size, hidden_size):
    super(Encoder, self).__init__(
        LSTM = links.LSTM(embed_size, hidden_size),
    )

  def __call__(self, x):
    """
    Updates the internal state and returns the RNN output (h).

    Args:
        x : A new batch from the input sequence.

    Returns:
        The output (h) of updated RNN unit.

    """
    return self.LSTM(x)

  def get_state():
    return self.LSTM.c

class StateLessEncoder(Chain):
  """
  This is just the same Encoder as below. The name is changed for the sake of disambiguation.
  The LSTM components are explicitly defined and the LSTM function is used in place of the LSTM class.

  Args: 
        embed_size - The size of embeddings of the inputs
        hidden_size - The size of the hidden layer representation of the RNN
        

  Attributes: 
        
  User Defined Methods:
  """
  def __init__(self, embed_size, hidden_size):
    super(Encoder, self).__init__(
        xh = links.Linear(embed_size, 4 * hidden_size),
        hh = links.Linear(hidden_size, 4 * hidden_size),
    )

  def __call__(self, x, c, h):
    """
    Updates the internal state and returns the RNN outputs for each layer as a list.

    Args:
        x : A new batch from the input sequence.
        c : The previous cell state.
        h : The previous cell output.
    Returns:
        The output (h) and the state (c) of the updated RNN unit.

    """
    return functions.lstm(c, self.xh(x) + self.hh(h))
    
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
    sum_e = XP.fzeros((batch_size, 1))
    for a, b in zip(a_list, b_list):
      w = functions.tanh(self.aw(a) + self.bw(b) + self.pw(p))
      e = functions.exp(self.we(w))
      e_list.append(e)
      sum_e += e
    ZEROS = XP.fzeros((batch_size, self.hidden_size))
    aa = ZEROS
    bb = ZEROS
    for a, b, e in zip(a_list, b_list, e_list):
      e /= sum_e
      aa += functions.reshape(functions.batch_matmul(a, e), (batch_size, self.hidden_size))
      bb += functions.reshape(functions.batch_matmul(b, e), (batch_size, self.hidden_size))
    return aa, bb

class LocalAttention(Chain):
  def __init__(self, hidden_size):
    super(Attention, self).__init__(
        aw = links.Linear(hidden_size, hidden_size),
        bw = links.Linear(hidden_size, hidden_size),
        pw = links.Linear(hidden_size, hidden_size),
        we = links.Linear(hidden_size, 1),
        ts = links.Linear(hidden_size, hidden_size),
        sp = links.Linear(hidden_size, 1),
    )
    self.hidden_size = hidden_size

  def __call__(self, a_list, b_list, p, sentence_length, window_size):
    batch_size = p.data.shape[0]
    SENTENCE_LENGTH = XP.fnonzeros((batch_size, 1),sentence_length)
    e_list = []
    sum_e = XP.fzeros((batch_size, 1))
    s = functions.tanh(self.ts(p))
    pos =  SENTENCE_LENGTH * functions.sigmoid(self.sp(s))

    # Develop batch logic to set to zero the components of a and b which are out of the window
    # Big question: Do I have to iterate over each element in the batch? That would suck.
    # One logic: Get global alignment matrix of (batch x) hidden size x sentence length and then another matrix of (batch x) sentence length which
    # will essentially be a matrix containing the gaussian distrubution weight and there will be zeros where the sentence position falls out of the window
    # Another logic: Create a matrix of (batch x) sentence length where there will be 1 for each position in the window

    # Separate the attention weights for a and b cause forward is different from backward.

    for a, b in zip(a_list, b_list):
      w = functions.tanh(self.aw(a) + self.bw(b) + self.pw(p))
      e = functions.exp(self.we(w))
      e_list.append(e)
      sum_e += e
    ZEROS = XP.fzeros((batch_size, self.hidden_size))
    aa = ZEROS
    bb = ZEROS
    for a, b, e in zip(a_list, b_list, e_list):
      e /= sum_e
      aa += a * e
      bb += b * e
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
    ZEROS = XP.fzeros((batch_size, self.hidden_size))
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

  x = XP.iarray([src_stoi('<s>') for _ in range(batch_size)])
  attmt.embed(x)
  for l in range(src_len):
    x = XP.iarray([src_stoi(src_batch[k][l]) for k in range(batch_size)])
    attmt.embed(x)
  x = XP.iarray([src_stoi('</s>') for _ in range(batch_size)])
  attmt.embed(x)

  attmt.encode()
  
  t = XP.iarray([trg_stoi('<s>') for _ in range(batch_size)])
  hyp_batch = [[] for _ in range(batch_size)]

  if is_training:
    loss = XP.fzeros(())
    for l in range(trg_len):
      y = attmt.decode(t)
      t = XP.iarray([trg_stoi(trg_batch[k][l]) for k in range(batch_size)])
      loss += functions.softmax_cross_entropy(y, t)
      output = cuda.to_cpu(y.data.argmax(1))
      for k in range(batch_size):
        hyp_batch[k].append(trg_itos(output[k]))
    return hyp_batch, loss
  
  else:
    while len(hyp_batch[0]) < generation_limit:
      y = attmt.decode(t)
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
  attmt = AttentionMT(args.vocab, args.embed, args.hidden)
  if args.use_gpu:
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
  if args.use_gpu:
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
  XP.set_library(args)
  if args.mode == 'train': train(args)
  elif args.mode == 'test': test(args)

if __name__ == '__main__':
  main()

