#!/usr/bin/python3

import my_settings

import datetime
import sys
import math
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict

from chainer import FunctionSet, Variable, functions, optimizers
from chainer import cuda


def trace(*args):
    print(datetime.datetime.now(), '...', *args, file=sys.stderr)
    sys.stderr.flush()


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
        #return Variable(np.array(array, dtype=dtype))
        return Variable(cuda.to_gpu(np.array(array, dtype=dtype)))

    @staticmethod
    def get_data(variable):
        #return variable.data
        return cuda.to_cpu(variable.data)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        #return Variable(np.zeros(shape, dtype=dtype))
        return Variable(cuda.zeros(shape, dtype=dtype))

    @staticmethod
    def make_model(**kwargs):
        #return FunctionSet(**kwargs)
        return FunctionSet(**kwargs).to_gpu()        
    

class EncoderDecoderModel:
    def __init__(self):
        pass

    def __new_opt(self):
        #return optimizers.SGD(lr=0.01)
        return optimizers.AdaGrad(lr=0.01)

    def __to_cpu(self):
        self.__model.to_cpu()
        self.__opt = self.__new_opt()
        self.__opt.setup(self.__model)

    def __to_gpu(self):
        self.__model.to_gpu()
        self.__opt = self.__new_opt()
        self.__opt.setup(self.__model)

    def __make_model(self):
        self.__model = wrapper.make_model(
            # encoder
            w_xi = functions.EmbedID(len(self.__src_vocab), self.__n_src_embed),
            w_ip = functions.Linear(self.__n_src_embed, 4 * self.__n_hidden),
            w_pp = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            # decoder
            w_pq = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            w_qj = functions.Linear(self.__n_hidden, self.__n_trg_embed),
            w_jy = functions.Linear(self.__n_trg_embed, len(self.__trg_vocab)),
            w_yq = functions.EmbedID(len(self.__trg_vocab), 4 * self.__n_hidden),
            w_qq = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
        )

        self.__to_gpu()

    @staticmethod
    def new(src_vocab, trg_vocab, n_src_embed, n_trg_embed, n_hidden):
        self = EncoderDecoderModel()
        
        self.__src_vocab = src_vocab
        self.__trg_vocab = trg_vocab
        self.__n_src_embed = n_src_embed
        self.__n_trg_embed = n_trg_embed
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

        self.__src_vocab.save(fp)
        self.__trg_vocab.save(fp)

        fprint(self.__n_src_embed)
        fprint(self.__n_trg_embed)
        fprint(self.__n_hidden)

        self.__to_cpu()
        
        print_embed(self.__model.w_xi)
        print_linear(self.__model.w_ip)
        print_linear(self.__model.w_pp)
        print_linear(self.__model.w_pq)
        print_linear(self.__model.w_qj)
        print_linear(self.__model.w_jy)
        print_embed(self.__model.w_yq)
        print_linear(self.__model.w_qq)

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

        self.__src_vocab = Vocabulary.load(line_gen)
        self.__trg_vocab = Vocabulary.load(line_gen)
        
        self.__n_src_embed = loadv(int)[0]
        self.__n_trg_embed = loadv(int)[0]
        self.__n_hidden = loadv(int)[0]

        self.__make_model()

        self.__to_cpu()

        copy_embed(self.__model.w_xi)
        copy_linear(self.__model.w_ip)
        copy_linear(self.__model.w_pp)
        copy_linear(self.__model.w_pq)
        copy_linear(self.__model.w_qj)
        copy_linear(self.__model.w_jy)
        copy_embed(self.__model.w_yq)
        copy_linear(self.__model.w_qq)

        self.__to_gpu()

        return self

    def __predict(self, is_training, src_batch, trg_batch = None, generation_limit = None):
        m = self.__model
        tanh = functions.tanh
        lstm = functions.lstm
        batch_size = len(src_batch)
        src_len = len(src_batch[0])
        src_stoi = self.__src_vocab.stoi
        trg_stoi = self.__trg_vocab.stoi
        trg_itos = self.__trg_vocab.itos

        s_c = wrapper.zeros((batch_size, self.__n_hidden))
        
        # encoding
        s_x = wrapper.make_var([src_stoi('</s>') for _ in range(batch_size)], dtype=np.int32)
        s_i = tanh(m.w_xi(s_x))
        s_c, s_p = lstm(s_c, m.w_ip(s_i))

        for l in reversed(range(src_len)):
            s_x = wrapper.make_var([src_stoi(src_batch[k][l]) for k in range(batch_size)], dtype=np.int32)
            s_i = tanh(m.w_xi(s_x))
            s_c, s_p = lstm(s_c, m.w_ip(s_i) + m.w_pp(s_p))

        s_c, s_q = lstm(s_c, m.w_pq(s_p))

        hyp_batch = [[] for _ in range(batch_size)]
        
        # decoding
        if is_training:
            accum_loss = wrapper.zeros(())
            trg_len = len(trg_batch[0])
            
            for l in range(trg_len):
                s_j = tanh(m.w_qj(s_q))
                r_y = m.w_jy(s_j)

                s_t = wrapper.make_var([trg_stoi(trg_batch[k][l]) for k in range(batch_size)], dtype=np.int32)
                accum_loss += functions.softmax_cross_entropy(r_y, s_t)
                
                output = wrapper.get_data(r_y).argmax(1)
                for k in range(batch_size):
                    hyp_batch[k].append(trg_itos(output[k]))

                #s_y = wrapper.make_var(output, dtype=np.int32)
                #s_c, s_q = lstm(s_c, m.w_yq(s_y) + m.w_qq(s_q))
                s_c, s_q = lstm(s_c, m.w_yq(s_t) + m.w_qq(s_q))

            return hyp_batch, accum_loss
        else:
            while len(hyp_batch[0]) < generation_limit:
                s_j = tanh(m.w_qj(s_q))
                r_y = m.w_jy(s_j)
                
                output = wrapper.get_data(r_y).argmax(1)
                for k in range(batch_size):
                    hyp_batch[k].append(trg_itos(output[k]))

                if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)): break

                s_y = wrapper.make_var(output, dtype=np.int32)
                s_c, s_q = lstm(s_c, m.w_yq(s_y) + m.w_qq(s_q))
            
            return hyp_batch

    def train(self, src_batch, trg_batch):
        self.__opt.zero_grads()

        hyp_batch, accum_loss = self.__predict(True, src_batch, trg_batch=trg_batch)
        
        accum_loss.backward()
        self.__opt.clip_grads(10)
        self.__opt.update()

        return hyp_batch, wrapper.get_data(accum_loss).reshape(())

    def predict(self, src_batch, generation_limit):
        return self.__predict(False, src_batch, generation_limit=generation_limit)


def parse_args():
    def_vocab = 32768
    def_embed_in = 256
    def_embed_out = 256
    def_hidden = 512
    def_epoch = 100
    def_minibatch = 256
    def_generation_limit = 256

    p = ArgumentParser(description='Encoder-decoder trainslation model trainer')

    p.add_argument('mode', help='\'train\' or \'test\'')
    p.add_argument('source', help='[in] source corpus')
    p.add_argument('target', help='[in/out] target corpus')
    p.add_argument('model', help='[in/out] model file')
    p.add_argument('--vocab', default=def_vocab, metavar='INT', type=int,
        help='vocabulary size (default: %d)' % def_vocab)
    p.add_argument('--embed-in', default=def_embed_in, metavar='INT', type=int,
        help='input embedding layer size (default: %d)' % def_embed_in)
    p.add_argument('--embed-out', default=def_embed_out, metavar='INT', type=int,
        help='output embedding layer size (default: %d)' % def_embed_out)
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
        if args.embed_in < 1: raise ValueError('you must set --embed-in >= 1')
        if args.embed_out < 1: raise ValueError('you must set --embed-out >= 1')
        if args.hidden < 1: raise ValueError('you must set --hidden >= 1')
        if args.epoch < 1: raise ValueError('you must set --epoch >= 1')
        if args.minibatch < 1: raise ValueError('you must set --minibatch >= 1')
        if args.generation_limit < 1: raise ValueError('you must set --generation-limit >= 1')
    except Exception as ex:
        p.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    return args


def generate_parallel_sorted(src_filename, trg_filename, batch_size):
    with open(src_filename) as fsrc, open(trg_filename) as ftrg:
        batch = []
        try:
            while True:
                for i in range(batch_size):
                    batch.append((next(fsrc).split(), next(ftrg).split()))
                batch = sorted(batch, key=lambda x: len(x[1]))
                for src, trg in batch:
                    yield src, trg
                batch = []
        except StopIteration:
            if batch:
                batch = sorted(batch, key=lambda x: len(x[1]))
                for src, trg in batch:
                    yield src, trg
        except:
            raise


def normalize_batch(batch):
    max_len = max(len(x) for x in batch)
    return [x + ['</s>'] * (max_len - len(x) + 1) for x in batch]


def generate_train_minibatch(src_filename, trg_filename, batch_size):
    gen = iter(generate_parallel_sorted(src_filename, trg_filename, batch_size * 100))
    src_batch = []
    trg_batch = []

    try:
        while True:
            for i in range(batch_size):
                src, trg = next(gen)
                src_batch.append(src)
                trg_batch.append(trg)
            
            yield normalize_batch(src_batch), normalize_batch(trg_batch)
            
            src_batch = []
            trg_batch = []
    except StopIteration:
        if src_batch and len(src_batch) == len(trg_batch):
            yield normalize_batch(src_batch), normalize_batch(trg_batch)
    except:
        raise


def generate_test_minibatch(src_filename, batch_size):
    with open(src_filename) as fsrc:
        batch = []
        try:
            while True:
                for i in range(batch_size):
                    batch.append(next(fsrc).split())
                yield normalize_batch(batch)
                batch = []
        except StopIteration:
            if batch:
                yield normalize_batch(batch)
        except:
            raise


def train_model(args):
    trace('making vocaburaries ...')
    with open(args.source) as fp: src_vocab = Vocabulary.new(fp, args.vocab)
    with open(args.target) as fp: trg_vocab = Vocabulary.new(fp, args.vocab)

    trace('start training ...')
    model = EncoderDecoderModel.new(src_vocab, trg_vocab, args.embed_in, args.embed_out, args.hidden)

    for epoch in range(args.epoch):
        trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
        log_ppl = 0.0
        trained = 0

        for src_batch, trg_batch in generate_train_minibatch(args.source, args.target, args.minibatch):
            hyp_batch, loss = model.train(src_batch, trg_batch)
            K = len(src_batch)

            for k in range(K):
                trace('epoch %3d/%3d, sample %8d' % (epoch + 1, args.epoch, trained + k + 1))
                trace('  src = ' + ' '.join([x if x != '</s>' else '*' for x in src_batch[k]]))
                trace('  trg = ' + ' '.join([x if x != '</s>' else '*' for x in trg_batch[k]]))
                trace('  hyp = ' + ' '.join([x if x != '</s>' else '*' for x in hyp_batch[k]]))

            trained += K

        with open(args.model + '.%03d' % (epoch + 1), 'w') as fp: model.save(fp)

    trace('finished.')


def test_model(args):
    trace('loading model ...')
    with open(args.model) as fp: model = EncoderDecoderModel.load(fp)

    trace('generating translation ...')
    generated = 0

    with open(args.target, 'w') as fp:
        for src_batch in generate_test_minibatch(args.source, args.minibatch):
            K = len(src_batch)
            trace('sample %8d - %8d ...' % (generated + 1, generated + K))

            hyp_batch = model.predict(src_batch, args.generation_limit)

            for hyp in hyp_batch:
                hyp.append('</s>')
                hyp = hyp[:hyp.index('</s>')]
                print(' '.join(hyp), file=fp)

            generated += K

    trace('finished.')


def main():
    args = parse_args()

    trace('initializing CUDA ...')
    cuda.init()

    if args.mode == 'train': train_model(args)
    elif args.mode == 'test': test_model(args)


if __name__ == '__main__':
    main()
