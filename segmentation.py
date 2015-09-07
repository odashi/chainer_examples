#!/usr/bin/python3

import datetime
import sys
import math
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

'''
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/appdirs-1.4.0-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/chainer-1.2.0-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/Mako-1.0.1-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/py-1.4.30-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/pycuda-2015.1.3-py3.4-linux-x86_64.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/pytest-2.7.2-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/pytools-2015.1.3-py3.4.egg')
sys.path.append('/project/nakamura-lab01/Work/yusuke-o/lib/python3.4/site-packages/scikit_cuda-0.5.0-py3.4.egg')
'''

from chainer import FunctionSet, Variable, functions, optimizers
#from chainer import cuda


def trace(*args):
    print(datetime.datetime.now(), '...', *args, file=sys.stderr)
    sys.stderr.flush()


def vocab_line_gen(filename):
    with open(filename) as fp:
        for l in fp:
            yield ' '.join(list(''.join(l.split())))


class Vocabulary:
    def __init__(self):
        pass

    def __len__(self):
        return self.__size

    def stoi(self, s):
        return self.__stoi[s]

    def itos(self, i):
        return self.__itos[i]

    @staticmethod
    def new(line_gen, size):
        self = Vocabulary()
        self.__size = size

        word_freq = defaultdict(lambda: 0)
        for line in line_gen:
            words = line.split()
            for word in words:
                word_freq[word] += 1

        self.__stoi = defaultdict(lambda: 0)
        self.__stoi['<unk>'] = 0
        self.__stoi['<s>'] = 1
        self.__stoi['</s>'] = 2
        self.__itos = [''] * self.__size
        self.__itos[0] = '<unk>'
        self.__itos[1] = '<s>'
        self.__itos[2] = '</s>'
        
        for i, (k, v) in zip(range(self.__size - 3), sorted(word_freq.items(), key=lambda x: -x[1])):
            self.__stoi[k] = i + 3
            self.__itos[i + 3] = k

        return self

    def save(self, fp):
        print(self.__size, file=fp)
        for i in range(self.__size):
            print(self.__itos[i], file=fp)

    @staticmethod
    def load(line_gen):
        self = Vocabulary()
        
        self.__size = int(next(line_gen))

        self.__stoi = defaultdict(lambda: 0)
        self.__itos = [''] * self.__size
        for i in range(self.__size):
            s = next(line_gen).strip()
            if s:
                self.__stoi[s] = i
                self.__itos[i] = s
        
        return self


class wrapper:
    @staticmethod
    def make_var(array, dtype=np.float32):
        return Variable(np.array(array, dtype=dtype))
        #return Variable(cuda.to_gpu(np.array(array, dtype=dtype)))

    @staticmethod
    def get_data(variable):
        return variable.data
        #return cuda.to_cpu(variable.data)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return Variable(np.zeros(shape, dtype=dtype))
        #return Variable(cuda.zeros(shape, dtype=dtype))

    @staticmethod
    def make_model(**kwargs):
        return FunctionSet(**kwargs)
        #return FunctionSet(**kwargs).to_gpu()        


class SegmentationModel:
    def __init__(self):
        pass

    def __new_opt(self):
        return optimizers.SGD(lr=0.01)
        #return optimizers.AdaGrad(lr=0.01)

    def __to_cpu(self):
        #self.__model.to_cpu()
        self.__opt = self.__new_opt()
        self.__opt.setup(self.__model)

    def __to_gpu(self):
        #self.__model.to_gpu()
        self.__opt = self.__new_opt()
        self.__opt.setup(self.__model)

    def __make_model(self):
        self.__model = wrapper.make_model(
            w_xh = functions.EmbedID(2 * self.__n_context * len(self.__vocab), self.__n_hidden),
            w_hy = functions.Linear(self.__n_hidden, 1),
        )
        self.__to_gpu()

    @staticmethod
    def new(vocab, n_context, n_hidden):
        self = SegmentationModel()
        self.__vocab = vocab
        self.__n_context = n_context
        self.__n_hidden = n_hidden
        self.__make_model()
        return self

    def save(self, fp):
        vtos = lambda v: ' '.join('%.8e' % x for x in v)
        fprint = lambda x: print(x, file=fp)

        def print_embed(f):
            for row in f.W: fprint(vtos(row))
        
        def print_linear(f):
            for row in f.W: fprint(vtos(row))
            fprint(vtos(f.b))

        self.__vocab.save(fp)
        fprint(self.__n_context)
        fprint(self.__n_hidden)
        self.__to_cpu()
        print_embed(self.__model.w_xh)
        print_linear(self.__model.w_hy)
        self.__to_gpu()

    @staticmethod
    def load(line_gen):
        loadv = lambda tp: [tp(x) for x in next(line_gen).split()]
        
        def copyv(tp, row):
            data = loadv(tp)
            for i in range(len(data)):
                row[i] = data[i]

        def copy_embed(f):
            for row in f.W: copyv(float, row)
            
        def copy_linear(f):
            for row in f.W: copyv(float, row)
            copyv(float, f.b)

        self = EncoderDecoderModel()
        self.__vocab = Vocabulary.load(line_gen)
        self.__n_context = loadv(int)[0]
        self.__n_hidden = loadv(int)[0]
        self.__make_model()
        self.__to_cpu()
        copy_embed(self.__model.w_xh)
        copy_linear(self.__model.w_hy)
        self.__to_gpu()

        return self

    def __make_training_data(self, text):
        words = text.split()
        c = self.__vocab.stoi
        m = self.__n_context - 1
        letters = [c('<s>')] * m + [c(x) for x in ''.join(words)] + [c('</s>')] * m
        labels = []
        for x in words:
            labels += [-1] * (len(x) - 1) + [1]
        return letters, labels[:-1]
        

    def train(self, text):
        m = self.__model
        tanh = functions.tanh

        self.__opt.zero_grads()

        letters, labels = self.__make_training_data(text)

        hyp = self.__vocab.itos(letters[-1 + self.__n_context])
        score = []
        accum_loss = wrapper.zeros(())
        
        for n in range(len(labels)):
            s_hu = wrapper.zeros((1, self.__n_hidden))
            for k in range(2 * self.__n_context):
                wid = k * len(self.__vocab) + letters[n + k]
                s_x = wrapper.make_var([wid], dtype=np.int32)
                s_hu += m.w_xh(s_x)
            s_hv = tanh(s_hu)
            s_y = tanh(m.w_hy(s_hv))
            s_t = wrapper.make_var([[labels[n]]])
            loss = functions.mean_squared_error(s_y, s_t)
            accum_loss += loss
            score.append(float(wrapper.get_data(s_y)))

            if score[-1] >= 0.0:
                hyp += ' '
            hyp += self.__vocab.itos(letters[n + self.__n_context])

        accum_loss.backward()
        self.__opt.clip_grads(5)
        self.__opt.update()

        return hyp, score

    def predict(self, text):
        pass


def parse_args():
    def_vocab = 3000
    def_hidden = 128
    def_epoch = 100
    def_context = 3

    p = ArgumentParser(description='Encoder-decoder trainslation model trainer')

    p.add_argument('mode', help='\'train\' or \'test\'')
    p.add_argument('corpus', help='[in] source corpus')
    p.add_argument('model', help='[in/out] model file')
    p.add_argument('--vocab', default=def_vocab, metavar='INT', type=int,
        help='vocabulary size (default: %d)' % def_vocab)
    p.add_argument('--hidden', default=def_hidden, metavar='INT', type=int,
        help='hidden layer size (default: %d)' % def_hidden)
    p.add_argument('--epoch', default=def_epoch, metavar='INT', type=int,
        help='number of training epoch (default: %d)' % def_epoch)
    p.add_argument('--context', default=def_context, metavar='INT', type=int,
        help='width of context window (default: %d)' % def_context)

    args = p.parse_args()

    # check args
    try:
        if args.mode not in ['train', 'test']: raise ValueError('you must set mode = \'train\' or \'test\'')
        if args.vocab < 1: raise ValueError('you must set --vocab >= 1')
        if args.hidden < 1: raise ValueError('you must set --hidden >= 1')
        if args.epoch < 1: raise ValueError('you must set --epoch >= 1')
        if args.context < 1: raise ValueError('you must set --context >= 1')
    except Exception as ex:
        p.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    return args


def train_model(args):
    trace('making vocaburaries ...')
    vocab = Vocabulary.new(vocab_line_gen(args.corpus), args.vocab)
    vocab.save(open('b', 'w'))

    trace('start training ...')
    model = SegmentationModel.new(vocab, args.context, args.hidden)

    for epoch in range(args.epoch):
        trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
        trained = 0

        with open(args.corpus) as fp:
            for text in fp:
                text = text.strip()
                hyp, score = model.train(text)
                trained += 1
                
                trace(trained)
                trace(text)
                trace(hyp)
                trace(' '.join('%+.1f' % x for x in score))
                
                if trained % 100 == 0:
                    trace('  %8d' % trained)

        trace('saveing model ...')
        with open(args.model + '.%03d' % (epoch + 1), 'w') as fp: model.save(fp)

    trace('finished.')


def test_model(args):
    pass


def main():
    args = parse_args()

    trace('initializing CUDA ...')
    #cuda.init()

    if args.mode == 'train': train_model(args)
    elif args.mode == 'test': test_model(args)


if __name__ == '__main__':
    main()
