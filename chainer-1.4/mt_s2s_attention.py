#!/usr/bin/python3

import my_settings

import sys
import math
import numpy as np
from argparse import ArgumentParser

from chainer import functions, optimizers
import chainer.computational_graph as cg

import util.generators as gens
from util.functions import trace, fill_batch2
from util.model_file import ModelFile
from util.vocabulary import Vocabulary

#from util.chainer_cpu_wrapper import wrapper
from util.chainer_gpu_wrapper import wrapper

   
class AttentionalTranslationModel:
    def __init__(self):
        pass

    def __make_model(self):
        self.__model = wrapper.make_model(
            # input embedding
            w_xi = functions.EmbedID(len(self.__src_vocab), self.__n_embed),
            # forward encoder
            w_ia = functions.Linear(self.__n_embed, 4 * self.__n_hidden),
            w_aa = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            # backward encoder
            w_ib = functions.Linear(self.__n_embed, 4 * self.__n_hidden),
            w_bb = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            # attentional weight estimator
            w_aw = functions.Linear(self.__n_hidden, self.__n_hidden),
            w_bw = functions.Linear(self.__n_hidden, self.__n_hidden),
            w_pw = functions.Linear(self.__n_hidden, self.__n_hidden),
            w_we = functions.Linear(self.__n_hidden, 1),
            # decoder
            w_ap = functions.Linear(self.__n_hidden, self.__n_hidden),
            w_bp = functions.Linear(self.__n_hidden, self.__n_hidden),
            w_yp = functions.EmbedID(len(self.__trg_vocab), 4 * self.__n_hidden),
            w_pp = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            w_cp = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            w_dp = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            w_py = functions.Linear(self.__n_hidden, len(self.__trg_vocab)),
        )

    @staticmethod
    def new(src_vocab, trg_vocab, n_embed, n_hidden):
        self = AttentionalTranslationModel()
        self.__src_vocab = src_vocab
        self.__trg_vocab = trg_vocab
        self.__n_embed = n_embed
        self.__n_hidden = n_hidden
        self.__make_model()
        return self

    def save(self, filename):
        with ModelFile(filename, 'w') as fp:
            self.__src_vocab.save(fp.get_file_pointer())
            self.__trg_vocab.save(fp.get_file_pointer())
            fp.write(self.__n_embed)
            fp.write(self.__n_hidden)
            wrapper.begin_model_access(self.__model)
            fp.write_embed(self.__model.w_xi)
            fp.write_linear(self.__model.w_ia)
            fp.write_linear(self.__model.w_aa)
            fp.write_linear(self.__model.w_ib)
            fp.write_linear(self.__model.w_bb)
            fp.write_linear(self.__model.w_aw)
            fp.write_linear(self.__model.w_bw)
            fp.write_linear(self.__model.w_pw)
            fp.write_linear(self.__model.w_we)
            fp.write_linear(self.__model.w_ap)
            fp.write_linear(self.__model.w_bp)
            fp.write_embed(self.__model.w_yp)
            fp.write_linear(self.__model.w_pp)
            fp.write_linear(self.__model.w_cp)
            fp.write_linear(self.__model.w_dp)
            fp.write_linear(self.__model.w_py)
            wrapper.end_model_access(self.__model)

    @staticmethod
    def load(filename):
        self = AttentionalTranslationModel()
        with ModelFile(filename) as fp:
            self.__src_vocab = Vocabulary.load(fp.get_file_pointer())
            self.__trg_vocab = Vocabulary.load(fp.get_file_pointer())
            self.__n_embed = int(fp.read())
            self.__n_hidden = int(fp.read())
            self.__make_model()
            wrapper.begin_model_access(self.__model)
            fp.read_embed(self.__model.w_xi)
            fp.read_linear(self.__model.w_ia)
            fp.read_linear(self.__model.w_aa)
            fp.read_linear(self.__model.w_ib)
            fp.read_linear(self.__model.w_bb)
            fp.read_linear(self.__model.w_aw)
            fp.read_linear(self.__model.w_bw)
            fp.read_linear(self.__model.w_pw)
            fp.read_linear(self.__model.w_we)
            fp.read_linear(self.__model.w_ap)
            fp.read_linear(self.__model.w_bp)
            fp.read_embed(self.__model.w_yp)
            fp.read_linear(self.__model.w_pp)
            fp.read_linear(self.__model.w_cp)
            fp.read_linear(self.__model.w_dp)
            fp.read_linear(self.__model.w_py)
            wrapper.end_model_access(self.__model)
        return self

    def init_optimizer(self):
        self.__opt = optimizers.AdaGrad(lr=0.01)
        self.__opt.setup(self.__model)

    def __forward(self, is_training, src_batch, trg_batch = None, generation_limit = None):
        m = self.__model
        tanh = functions.tanh
        lstm = functions.lstm
        batch_size = len(src_batch)
        hidden_size = self.__n_hidden
        src_len = len(src_batch[0])
        trg_len = len(trg_batch[0]) - 1 if is_training else generation_limit
        src_stoi = self.__src_vocab.stoi
        trg_stoi = self.__trg_vocab.stoi
        trg_itos = self.__trg_vocab.itos

        hidden_zeros = wrapper.zeros((batch_size, hidden_size))
        sum_e_zeros = wrapper.zeros((batch_size, 1))

        # make embedding
        list_x = []
        for l in range(src_len):
            s_x = wrapper.make_var([src_stoi(src_batch[k][l]) for k in range(batch_size)], dtype=np.int32)
            list_x.append(s_x)

        # forward encoding
        c = hidden_zeros
        s_a = hidden_zeros
        list_a = []
        for l in range(src_len):
            s_x = list_x[l]
            s_i = tanh(m.w_xi(s_x))
            c, s_a = lstm(c, m.w_ia(s_i) + m.w_aa(s_a))
            list_a.append(s_a)
        
        # backward encoding
        c = hidden_zeros
        s_b = hidden_zeros
        list_b = []
        for l in reversed(range(src_len)):
            s_x = list_x[l]
            s_i = tanh(m.w_xi(s_x))
            c, s_b = lstm(c, m.w_ib(s_i) + m.w_bb(s_b))
            list_b.insert(0, s_b)

        # decoding
        c = hidden_zeros
        s_p = tanh(m.w_ap(list_a[-1]) + m.w_bp(list_b[0]))
        s_y = wrapper.make_var([trg_stoi('<s>') for k in range(batch_size)], dtype=np.int32)

        hyp_batch = [[] for _ in range(batch_size)]
        accum_loss = wrapper.zeros(()) if is_training else None
        
        #for n in range(src_len):
        #    print(src_batch[0][n], end=' ')
        #print()

        for l in range(trg_len):
            # calculate attention weights
            list_e = []
            sum_e = sum_e_zeros
            for n in range(src_len):
                s_w = tanh(m.w_aw(list_a[n]) + m.w_bw(list_b[n]) + m.w_pw(s_p))
                r_e = functions.exp(m.w_we(s_w))
                #list_e.append(functions.concat(r_e for _ in range(self.__n_hidden)))
                list_e.append(r_e)
                sum_e += r_e
            #sum_e = functions.concat(sum_e for _ in range(self.__n_hidden))

            # make attention vector
            s_c = hidden_zeros
            s_d = hidden_zeros
            for n in range(src_len):
                s_e = list_e[n] / sum_e
                #s_c += s_e * list_a[n]
                #s_d += s_e * list_b[n]
                s_c += functions.reshape(functions.batch_matmul(list_a[n], s_e), (batch_size, hidden_size))
                s_d += functions.reshape(functions.batch_matmul(list_b[n], s_e), (batch_size, hidden_size))

                #zxcv = wrapper.get_data(s_e)[0][0]
                #if zxcv > 0.9: asdf='#'
                #elif zxcv > 0.7: asdf='*'
                #elif zxcv > 0.3: asdf='+'
                #elif zxcv > 0.1: asdf='.'
                #else: asdf=' '
                #print(asdf * len(src_batch[0][n]), end=' ')

            # generate next word
            c, s_p = lstm(c, m.w_yp(s_y) + m.w_pp(s_p) + m.w_cp(s_c) + m.w_dp(s_d))
            r_y = m.w_py(s_p)
            output = wrapper.get_data(r_y).argmax(1)
            for k in range(batch_size):
                hyp_batch[k].append(trg_itos(output[k]))

            #print(hyp_batch[0][-1])
            
            if is_training:
                s_t = wrapper.make_var([trg_stoi(trg_batch[k][l + 1]) for k in range(batch_size)], dtype=np.int32)
                accum_loss += functions.softmax_cross_entropy(r_y, s_t)
                s_y = s_t
            else:
                if all(hyp_batch[k][-1] == '</s>' for k in range(batch_size)): break
                s_y = wrapper.make_var(output, dtype=np.int32)

        return hyp_batch, accum_loss

    def train(self, src_batch, trg_batch):
        self.__opt.zero_grads()
        hyp_batch, accum_loss = self.__forward(True, src_batch, trg_batch=trg_batch)
        #g = cg.build_computational_graph([accum_loss])
        #with open('asdf', 'w') as fp: fp.write(g.dump())
        #sys.exit()
        accum_loss.backward()
        self.__opt.clip_grads(10)
        self.__opt.update()
        return hyp_batch

    def predict(self, src_batch, generation_limit):
        return self.__forward(False, src_batch, generation_limit=generation_limit)[0]


def parse_args():
    def_vocab = 32768
    def_embed = 256
    def_hidden = 512
    def_epoch = 100
    def_minibatch = 64
    def_generation_limit = 256

    p = ArgumentParser(description='Attentional neural machine translation')

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


def train_model(args):
    trace('making vocabularies ...')
    src_vocab = Vocabulary.new(gens.word_list(args.source), args.vocab)
    trg_vocab = Vocabulary.new(gens.word_list(args.target), args.vocab)

    trace('making model ...')
    model = AttentionalTranslationModel.new(src_vocab, trg_vocab, args.embed, args.hidden)

    for epoch in range(args.epoch):
        trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
        trained = 0
        gen1 = gens.word_list(args.source)
        gen2 = gens.word_list(args.target)
        gen3 = gens.batch(gens.sorted_parallel(gen1, gen2, 100 * args.minibatch, order=0), args.minibatch)
        model.init_optimizer()

        for src_batch, trg_batch in gen3:
            src_batch = fill_batch2(src_batch)
            trg_batch = fill_batch2(trg_batch)
            K = len(src_batch)
            hyp_batch = model.train(src_batch, trg_batch)

            for k in range(K):
                trace('epoch %3d/%3d, sample %8d' % (epoch + 1, args.epoch, trained + k + 1))
                trace('  src = ' + ' '.join([x if x != '</s>' else '*' for x in src_batch[k]]))
                trace('  trg = ' + ' '.join([x if x != '</s>' else '*' for x in trg_batch[k]]))
                trace('  hyp = ' + ' '.join([x if x != '</s>' else '*' for x in hyp_batch[k]]))

            trained += K

        trace('saving model ...')
        model.save(args.model + '.%03d' % (epoch + 1))

    trace('finished.')


def test_model(args):
    trace('loading model ...')
    model = AttentionalTranslationModel.load(args.model)
    
    trace('generating translation ...')
    generated = 0

    with open(args.target, 'w') as fp:
        for src_batch in gens.batch(gens.word_list(args.source), args.minibatch):
            src_batch = fill_batch2(src_batch)
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

    trace('initializing ...')
    wrapper.init()

    if args.mode == 'train': train_model(args)
    elif args.mode == 'test': test_model(args)


if __name__ == '__main__':
    main()

