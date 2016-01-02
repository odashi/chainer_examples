#!/usr/bin/python3

import my_settings

import sys
import math
import numpy as np
from argparse import ArgumentParser

from chainer import functions, optimizers

import util.generators as gens
from util.functions import trace, fill_batch
from util.model_file import ModelFile
from util.vocabulary import Vocabulary

#from util.chainer_cpu_wrapper import wrapper
from util.chainer_gpu_wrapper import wrapper

   
class EncoderDecoderModel:
    def __init__(self):
        pass

    def __make_model(self):
        self.__model = wrapper.make_model(
            # encoder
            w_xi = functions.EmbedID(len(self.__src_vocab), self.__n_embed),
            w_ip = functions.Linear(self.__n_embed, 4 * self.__n_hidden),
            w_pp = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            # decoder
            w_pq = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            w_qj = functions.Linear(self.__n_hidden, self.__n_embed),
            w_jy = functions.Linear(self.__n_embed, len(self.__trg_vocab)),
            w_yq = functions.EmbedID(len(self.__trg_vocab), 4 * self.__n_hidden),
            w_qq = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
        )

    @staticmethod
    def new(src_vocab, trg_vocab, n_embed, n_hidden):
        self = EncoderDecoderModel()
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
            fp.write_linear(self.__model.w_ip)
            fp.write_linear(self.__model.w_pp)
            fp.write_linear(self.__model.w_pq)
            fp.write_linear(self.__model.w_qj)
            fp.write_linear(self.__model.w_jy)
            fp.write_embed(self.__model.w_yq)
            fp.write_linear(self.__model.w_qq)
            wrapper.end_model_access(self.__model)

    @staticmethod
    def load(filename):
        self = EncoderDecoderModel()
        with ModelFile(filename) as fp:
            self.__src_vocab = Vocabulary.load(fp.get_file_pointer())
            self.__trg_vocab = Vocabulary.load(fp.get_file_pointer())
            self.__n_embed = int(fp.read())
            self.__n_hidden = int(fp.read())
            self.__make_model()
            wrapper.begin_model_access(self.__model)
            fp.read_embed(self.__model.w_xi)
            fp.read_linear(self.__model.w_ip)
            fp.read_linear(self.__model.w_pp)
            fp.read_linear(self.__model.w_pq)
            fp.read_linear(self.__model.w_qj)
            fp.read_linear(self.__model.w_jy)
            fp.read_embed(self.__model.w_yq)
            fp.read_linear(self.__model.w_qq)
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
        hyp_batch, accum_loss = self.__forward(True, src_batch, trg_batch=trg_batch)
        accum_loss.backward()
        self.__opt.clip_grads(10)
        self.__opt.update()
        return hyp_batch

    def predict(self, src_batch, generation_limit):
        return self.__forward(False, src_batch, generation_limit=generation_limit)


def parse_args():
    def_vocab = 32768
    def_embed = 256
    def_hidden = 512
    def_epoch = 100
    def_minibatch = 64
    def_generation_limit = 256

    p = ArgumentParser(description='Encoder-decoder neural machine trainslation')

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
    model = EncoderDecoderModel.new(src_vocab, trg_vocab, args.embed, args.hidden)

    for epoch in range(args.epoch):
        trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
        trained = 0
        gen1 = gens.word_list(args.source)
        gen2 = gens.word_list(args.target)
        gen3 = gens.batch(gens.sorted_parallel(gen1, gen2, 100 * args.minibatch), args.minibatch)
        model.init_optimizer()

        for src_batch, trg_batch in gen3:
            src_batch = fill_batch(src_batch)
            trg_batch = fill_batch(trg_batch)
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
    model = EncoderDecoderModel.load(args.model)
    
    trace('generating translation ...')
    generated = 0

    with open(args.target, 'w') as fp:
        for src_batch in gens.batch(gens.word_list(args.source), args.minibatch):
            src_batch = fill_batch(src_batch)
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

