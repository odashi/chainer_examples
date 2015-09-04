#!/usr/bin/python3

import datetime
import sys
import math
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/appdirs-1.4.0-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/chainer-1.2.0-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/Mako-1.0.1-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/py-1.4.30-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/pycuda-2015.1.3-py3.4-linux-x86_64.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/pytest-2.7.2-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/pytools-2015.1.3-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/scikit_cuda-0.5.0-py3.4.egg')

from chainer import FunctionSet, Variable, cuda, functions, optimizers


def trace(text):
    print(datetime.datetime.now(), '...', text, file=sys.stderr)


def make_var(array, dtype=np.float32):
    #return Variable(np.array(array, dtype=dtype))
    return Variable(cuda.to_gpu(np.array(array, dtype=dtype)))

def get_data(variable):
    #return variable.data
    return cuda.to_cpu(variable.data)

def zeros(shape, dtype=np.float32):
    #return Variable(np.zeros(shape, dtype=dtype))
    return Variable(cuda.zeros(shape, dtype=dtype))

def make_model(**kwargs):
    #return FunctionSet(**kwargs)
    return FunctionSet(**kwargs).to_gpu()


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


def make_rnnlm_model(n_vocab, n_embed, n_hidden):
    return make_model(
        w_xe = functions.EmbedID(n_vocab, n_embed),
        w_eh = functions.Linear(n_embed, n_hidden),
        w_hh = functions.Linear(n_hidden, n_hidden),
        w_hy = functions.Linear(n_hidden, n_vocab),
    )


def save_rnnlm_model(filename, n_vocab, n_embed, n_hidden, vocab, model):
    fmt = '%.8e'
    dlm = ' '

    model.to_cpu()

    with open(filename, 'w') as fp:
        print(n_vocab, file=fp)
        print(n_embed, file=fp)
        print(n_hidden, file=fp)

        for k, v in vocab.items():
            if v == 0:
                continue
            print('%s %d' % (k, v), file=fp)
        
        for row in model.w_xe.W:
            print(dlm.join(fmt % x for x in row), file=fp)
        
        for row in model.w_eh.W:
            print(dlm.join(fmt % x for x in row), file=fp)
        print(dlm.join(fmt % x for x in model.w_eh.b), file=fp)
        
        for row in model.w_hh.W:
            print(dlm.join(fmt % x for x in row), file=fp)
        print(dlm.join(fmt % x for x in model.w_hh.b), file=fp)
        
        for row in model.w_hy.W:
            print(dlm.join(fmt % x for x in row), file=fp)
        print(dlm.join(fmt % x for x in model.w_hy.b), file=fp)
    
    model.to_gpu()


def parse_args():
    def_vocab = 40000
    def_embed = 200
    def_hidden = 200
    def_epoch = 10
    def_minibatch = 256

    p = ArgumentParser(description='RNNLM trainer')

    p.add_argument('corpus', help='[in] training corpus')
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
        print(ex, file=sys.stderr)
        sys.exit()

    return args

        
def main():
    args = parse_args()

    trace('making vocaburary ...')
    vocab, num_lines, num_words = make_vocab(args.corpus, args.vocab)

    trace('initializing CUDA ...')
    cuda.init()

    trace('start training ...')
    model = make_rnnlm_model(args.vocab, args.embed, args.hidden)

    for epoch in range(args.epoch):
        trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
        log_ppl = 0.0
        trained = 0
        
        opt = optimizers.SGD()
        opt.setup(model)

        for batch in generate_batch(args.corpus, args.minibatch):
            batch = [[vocab[x] for x in words] for words in batch]
            K = len(batch)
            L = len(batch[0]) - 1

            opt.zero_grads()
            s_h = zeros((K, args.hidden))

            for l in range(L):
                s_x = make_var([batch[k][l] for k in range(K)], dtype=np.int32)
                s_t = make_var([batch[k][l + 1] for k in range(K)], dtype=np.int32)

                s_e = functions.tanh(model.w_xe(s_x))
                s_h = functions.tanh(model.w_eh(s_e) + model.w_hh(s_h))
                s_y = model.w_hy(s_h)

                loss = functions.softmax_cross_entropy(s_y, s_t)
                loss.backward()
            
                log_ppl += get_data(loss).reshape(()) * K

            opt.update()
            trained += K
            trace('  %d/%d' % (trained, num_lines))
            
        log_ppl /= float(num_words)
        trace('  log(PPL) = %.10f' % log_ppl)
        trace('  PPL      = %.10f' % math.exp(log_ppl))

        trace('  writing model ...')
        save_rnnlm_model(args.model + '.%d' % (epoch + 1), args.vocab, args.embed, args.hidden, vocab, model)

    trace('training finished.')


if __name__ == '__main__':
    main()

