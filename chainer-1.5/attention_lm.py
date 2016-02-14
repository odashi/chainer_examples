import sys
import numpy
from argparse import ArgumentParser
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import util.generators as gens
from util.functions import trace, fill_batch
from util.vocabulary import Vocabulary


#Added comment

def make_vocab(filename, vocab_size):
    word_freq = defaultdict(lambda: 0)
    num_lines = 0
    num_words = 0
    with open(filename) as fp:
        for line in fp:
            words = line.split()
            num_lines += 1
            num_words += len(words)
            for word in words:
                word_freq[word] += 1

    # 0: unk
    # 1: <s>
    # 2: </s>
    vocab = defaultdict(lambda: 0)
    vocab['<s>'] = 1
    vocab['</s>'] = 2
    for i,(k,v) in zip(range(vocab_size - 3), sorted(word_freq.items(), key=lambda x: -x[1])):
        vocab[k] = i + 3

    return vocab, num_lines, num_words


def generate_batch(filename, batch_size):
    with open(filename) as fp:
        batch = []
        try:
            while True:
                for i in range(batch_size):
                    batch.append(next(fp).split())
                
                max_len = max(len(x) for x in batch)
                batch = [['<s>'] + x + ['</s>'] * (max_len - len(x) + 1) for x in batch]
                yield batch
                
                batch = []
        except:
            pass

        if batch:
            max_len = max(len(x) for x in batch)
            batch = [['<s>'] + x + ['</s>'] * (max_len - len(x) + 1) for x in batch]
            yield batch



def get_data(variable):
    #return variable.data
    return cuda.to_cpu(variable.data)

def parse_args():
    def_vocab = 40000
    def_embed = 200
    def_hidden = 200
    def_epoch = 10
    def_minibatch = 256
    def_model = 0
    p = ArgumentParser(description='RNNLM trainer')

    p.add_argument('corpus', help='[in] training corpus')
    p.add_argument('valid', help='[in] validation corpus')
    p.add_argument('model', help='[out] model file')
    p.add_argument('-V', '--vocab', default=def_vocab, metavar='INT', type=int,
        help='vocabulary size (default: %d)' % def_vocab)
    p.add_argument('-E', '--embed', default=def_embed, metavar='INT', type=int,
        help='embedding layer size (default: %d)' % def_embed)
    p.add_argument('-H', '--hidden', default=def_hidden, metavar='INT', type=int,
        help='hidden layer size (default: %d)' % def_hidden)
    p.add_argument('-I', '--epoch', default=def_epoch, metavar='INT', type=int,
        help='number of training epoch (default: %d)' % def_epoch)
    p.add_argument('-B', '--minibatch', default=def_minibatch, metavar='INT', type=int,
        help='minibatch size (default: %d)' % def_minibatch)
    p.add_argument('-M', '--model', default=def_model, metavar='INT', type=int,
        help='RNN used for LM (default: %d) where 0: Default RNNLM, 1: LSTM RNNLM, 2: Attention RNNLM' % def_model)

    args = p.parse_args()

    # check args
    try:
        if (args.vocab < 1): raise ValueError('you must set --vocab >= 1')
        if (args.embed < 1): raise ValueError('you must set --embed >= 1')
        if (args.hidden < 1): raise ValueError('you must set --hidden >= 1')
        if (args.epoch < 1): raise ValueError('you must set --epoch >= 1')
        if (args.minibatch < 1): raise ValueError('you must set --minibatch >= 1')
    except Exception as ex:
        p.print_usage(file=sys.stderr)
        print(ex)
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

class BasicRnnLM(Chain):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(BasicRnn, self).__init__(
            xe = SrcEmbed(vocab_size, embed_size),
            eh = links.Linear(embed_size, hidden_size),
            hh = links.Linear(hidden_size, hidden_size),
            hy = links.Linear(hidden_size, vocab_size),
        )
    self.reset_state()

    def reset_state():
        self.h = None

    def __call__(self, x):

        e = self.xe(x)
        h = self.eh(e)
        if self.h is not None:
            h += self.hh(self.h)
        self.h = functions.tanh(h)
        y = self.hy(self.h)
        return y

class LSTMLM(Chain):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(LSTMRnn, self).__init__(
            xe = SrcEmbed(vocab_size, embed_size),
            lstm = links.LSTM(embed_size, hidden_size),
            hy = links.Linear(hidden_size, vocab_size),
        )

    def reset(self):
        self.zerograds()

    def __call__(self, x):
        e = self.xe(x)
        h = self.lstm(e)
        y = self.hy(h)
        return y

class LSTMEncoder(Chain):
    def __init__(self, embed_size, hidden_size):
        super(LSTMEncoder, self).__init__(
            lstm = links.LSTM(embed_size, hidden_size),
        )
    def reset(self):
        self.zerograds()
    def __call__(self, x):
        h = self.lstm(x)
        return h

class Attention(Chain):
  def __init__(self, hidden_size, embed_size):
    super(Attention, self).__init__(
        aw = links.Linear(embed_size, hidden_size),
        pw = links.Linear(hidden_size, hidden_size),
        we = links.Linear(hidden_size, 1),
    )
    self.hidden_size = hidden_size
  
  
    
  def __call__(self, a_list, p):
    batch_size = p.data.shape[0]
    e_list = []
    sum_e = XP.fzeros((batch_size, 1))
    for a in a_list:
      w = functions.tanh(self.aw(a) + self.pw(p))
      e = functions.exp(self.we(w))
      e_list.append(e)
      sum_e += e
    ZEROS = XP.fzeros((batch_size, self.hidden_size))
    aa = ZEROS
    for a, e in zip(a_list, e_list):
      e /= sum_e
      aa += a * e
      #aa += functions.reshape(functions.batch_matmul(a, e), (batch_size, self.hidden_size))
    return aa

class AttentionLM(Chain):
  def __init__(self, embed_size, hidden_size, vocab_size):
    super(AttentionMT, self).__init__(
        emb = SrcEmbed(vocab_size, embed_size),
        enc = LSTMEncoder(embed_size, hidden_size),
        att = Attention(hidden_size, embed_size),
        outhe = links.Linear(hidden_size, hidden_size),
        outae = links.Linear(hidden_size, hidden_size),
        outey = links.Linear(hidden_size, vocab_size),
    )
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size

  def reset(self):
    self.zerograds()
    self.x_list = []

  def embed(self, x):
    self.x_list.append(self.emb(x))

  def encode(self, x):
    self.h = self.enc(x)

  def decode(self, atts_list):
    aa = self.att(self.atts_list, self.h)
    y = tanh(self.outhe(self.h) + self.outae(aa))
    return self.outey(y)

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
      return AttentionLM(embed_size, hidden_size, vocab_size)

def forward(batch, model):
    batch = [[vocab[x] for x in words] for words in batch]
    K = len(batch)
    L = len(batch[0]) - 1

    opt.zero_grads()
    accum_loss = XP.fzeros(())
    accum_log_ppl = XP.fzeros(())

    if args.model is 0 or args.model is 1:
        
        for l in range(L):
            s_x = make_var([batch[k][l] for k in range(K)], dtype=np.int32)
            s_t = make_var([batch[k][l + 1] for k in range(K)], dtype=np.int32)

            s_y = model(s_x)

            loss_i = functions.softmax_cross_entropy(s_y, s_t)
            accum_loss += loss_i
        
            accum_log_ppl += get_data(loss_i)

        

    else:
        for l in range(L):
            s_x = make_var([batch[k][l] for k in range(K)], dtype=np.int32)
            model.embed(s_x)
        for l in range(L):
            s_t = make_var([batch[k][l + 1] for k in range(K)], dtype=np.int32)
            model.encode(self.x_list[l])
            s_y = model.decode(self.x_list[0:l]+self.x_list[l+1:L])
            
            loss_i = functions.softmax_cross_entropy(s_y, s_t)
            accum_loss += loss_i
        
            accum_log_ppl += get_data(loss_i)
    
    return accum_loss, accum_log_ppl
            

def main():
    args = parse_args()

    trace('making vocabulary ...')
    vocab, num_lines, num_words = make_vocab(args.corpus, args.vocab)

    trace('initializing CUDA ...')
    cuda.init()

    trace('start training ...')
    if args.model is 0:
        model = BasicRnnLM(args.embed, args.hidden, args.vocab)
        model.reset()
    elif args.model is 1:
        model = LSTMRnn(args.embed, args.hidden, args.vocab)
        model.reset()
    elif args.model is 2:
        model = AttentionLM(args.embed, args.hidden, args.vocab)
        model.reset()
    model.to_gpu()

    for epoch in range(args.epoch):
        trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
        log_ppl = 0.0
        trained = 0
        
        opt = optimizers.AdaGrad(lr = 0.01)
        opt.setup(model)
        opt.add_hook(optimizer.GradientClipping(5))

        for batch in generate_batch(args.corpus, args.minibatch):
            K = len(batch)
            loss, perplexity= forward(batch, model)
            loss.backward()
            log_ppl += perplexity 
            opt.update()
            trained += K
            model.reset()

        trace('  %d/%d' % (trained, num_lines))      
        log_ppl /= float(num_words)
        trace('Train  log(PPL) = %.10f' % log_ppl)
        trace('Train  PPL      = %.10f' % math.exp(log_ppl))

        log_ppl = 0.0

        for batch in generate_batch(args.valid, args.minibatch):
            K = len(batch)
            loss, perplexity= forward(batch, model)
            log_ppl += perplexity 
            model.reset()

        trace('Valid  log(PPL) = %.10f' % log_ppl)
        trace('Valid  PPL      = %.10f' % math.exp(log_ppl))

        trace('  writing model ...')
        trace('saving model ...')
        prefix = 'RNNLM-'+str(args.model) + '.%03.d' % (epoch + 1)
        save_vocab(prefix + '.srcvocab',vocab) #Fix this # Fixed
        model.save_spec(prefix + '.spec')
        serializers.save_hdf5(prefix + '.weights', model)

    trace('training finished.')


if __name__ == '__main__':
    main()


def save_vocab(filename, vocab):
    with open(filename, 'w') as fp:
        for k, v in vocab.items():
            if v == 0:
                continue
            print('%s %d' % (k, v), file=fp)
