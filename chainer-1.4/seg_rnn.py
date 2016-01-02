#!/usr/bin/python3

#import my_settings

import sys
import math
import numpy as np
from argparse import ArgumentParser

from chainer import functions, optimizers

import util.generators as gens
from util.functions import trace, fill_batch
from util.model_file import ModelFile
from util.vocabulary import Vocabulary

from util.chainer_cpu_wrapper import wrapper
#from util.chainer_gpu_wrapper import wrapper


class RNNSegmentationModel:
    def __init__(self):
        pass

    def __make_model(self):
        self.__model = wrapper.make_model(
            w_xe = functions.EmbedID(len(self.__vocab), self.__n_embed),
            w_ea = functions.Linear(self.__n_embed, 4 * self.__n_hidden),
            w_aa = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            w_eb = functions.Linear(self.__n_embed, 4 * self.__n_hidden),
            w_bb = functions.Linear(self.__n_hidden, 4 * self.__n_hidden),
            w_ay1 = functions.Linear(self.__n_hidden, 1),
            w_by1 = functions.Linear(self.__n_hidden, 1),
            w_ay2 = functions.Linear(self.__n_hidden, 1),
            w_by2 = functions.Linear(self.__n_hidden, 1),
        )

    @staticmethod
    def new(vocab, n_embed, n_hidden):
        self = RNNSegmentationModel()
        self.__vocab = vocab
        self.__n_embed = n_embed
        self.__n_hidden = n_hidden
        self.__make_model()
        return self

    def save(self, filename):
        with ModelFile(filename, 'w') as fp:
            self.__vocab.save(fp.get_file_pointer())
            fp.write(self.__n_embed)
            fp.write(self.__n_hidden)
            wrapper.begin_model_access(self.__model)
            fp.write_embed(self.__model.w_xe)
            fp.write_linear(self.__model.w_ea)
            fp.write_linear(self.__model.w_aa)
            fp.write_linear(self.__model.w_eb)
            fp.write_linear(self.__model.w_bb)
            fp.write_linear(self.__model.w_ay1)
            fp.write_linear(self.__model.w_by1)
            fp.write_linear(self.__model.w_ay2)
            fp.write_linear(self.__model.w_by2)
            wrapper.end_model_access(self.__model)

    @staticmethod
    def load(filename):
        self = RNNSegmentationModel()
        with ModelFile(filename) as fp:
            self.__vocab = Vocabulary.load(fp.get_file_pointer())
            self.__n_embed = int(fp.read())
            self.__n_hidden = int(fp.read())
            self.__make_model()
            wrapper.begin_model_access(self.__model)
            fp.read_embed(self.__model.w_xe)
            fp.read_linear(self.__model.w_ea)
            fp.read_linear(self.__model.w_aa)
            fp.read_linear(self.__model.w_eb)
            fp.read_linear(self.__model.w_bb)
            fp.read_linear(self.__model.w_ay1)
            fp.read_linear(self.__model.w_by1)
            fp.read_linear(self.__model.w_ay2)
            fp.read_linear(self.__model.w_by2)
            wrapper.end_model_access(self.__model)
        return self

    def init_optimizer(self):
        self.__opt = optimizers.AdaGrad(lr=0.001)
        self.__opt.setup(self.__model)

    def __make_input(self, is_training, text):
        word_list = text.split()
        letters = [self.__vocab.stoi(x) for x in ''.join(word_list)]
        if is_training:
            labels = []
            for x in word_list:
                labels += [-1] * (len(x) - 1) + [1]
            return letters, labels[:-1]
        else:
            return letters, None

    def __forward(self, is_training, text):
        m = self.__model
        tanh = functions.tanh
        lstm = functions.lstm
        letters, labels = self.__make_input(is_training, text)
        n_letters = len(letters)

        accum_loss = wrapper.zeros(()) if is_training else None
        hidden_zeros = wrapper.zeros((1, self.__n_hidden))

        # embedding
        list_e = []
        for i in range(n_letters):
            s_x = wrapper.make_var([letters[i]], dtype=np.int32)
            list_e.append(tanh(m.w_xe(s_x)))

        # forward encoding
        s_a = hidden_zeros
        c = hidden_zeros
        list_a = []
        for i in range(n_letters):
            c, s_a = lstm(c, m.w_ea(list_e[i]) + m.w_aa(s_a))
            list_a.append(s_a)
        
        # backward encoding
        s_b = hidden_zeros
        c = hidden_zeros
        list_b = []
        for i in reversed(range(n_letters)):
            c, s_b = lstm(c, m.w_eb(list_e[i]) + m.w_bb(s_b))
            list_b.append(s_b)
        
        # segmentation
        scores = []
        for i in range(n_letters - 1):
            s_y = tanh(m.w_ay1(list_a[i]) + m.w_by1(list_b[i]) + m.w_ay2(list_a[i + 1]) + m.w_by2(list_b[i + 1]))
            scores.append(float(wrapper.get_data(s_y)))
            
            if is_training:
                s_t = wrapper.make_var([[labels[i]]])
                accum_loss += functions.mean_squared_error(s_y, s_t)

        return scores, accum_loss

    def train(self, text):
        self.__opt.zero_grads()
        scores, accum_loss = self.__forward(True, text)
        accum_loss.backward()
        self.__opt.clip_grads(5)
        self.__opt.update()
        return scores

    def predict(self, text):
        return self.__forward(False, text)[0]


def parse_args():
    def_vocab = 2500
    def_embed = 100
    def_hidden = 100
    def_epoch = 20

    p = ArgumentParser(description='Word segmentation using LSTM-RNN')

    p.add_argument('mode', help='\'train\' or \'test\'')
    p.add_argument('corpus', help='[in] source corpus')
    p.add_argument('model', help='[in/out] model file')
    p.add_argument('--vocab', default=def_vocab, metavar='INT', type=int,
        help='vocabulary size (default: %d)' % def_vocab)
    p.add_argument('--embed', default=def_embed, metavar='INT', type=int,
        help='embedding layer size (default: %d)' % def_embed)
    p.add_argument('--hidden', default=def_hidden, metavar='INT', type=int,
        help='hidden layer size (default: %d)' % def_hidden)
    p.add_argument('--epoch', default=def_epoch, metavar='INT', type=int,
        help='number of training epoch (default: %d)' % def_epoch)

    args = p.parse_args()

    # check args
    try:
        if args.mode not in ['train', 'test']: raise ValueError('you must set mode = \'train\' or \'test\'')
        if args.vocab < 1: raise ValueError('you must set --vocab >= 1')
        if args.embed < 1: raise ValueError('you must set --embed >= 1')
        if args.hidden < 1: raise ValueError('you must set --hidden >= 1')
        if args.epoch < 1: raise ValueError('you must set --epoch >= 1')
    except Exception as ex:
        p.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    return args


def make_hyp(letters, scores):
    hyp = letters[0]
    for w, s in zip(letters[1:], scores):
        if s >= 0:
            hyp += ' '
        hyp += w
    return hyp


def train_model(args):
    trace('making vocabularies ...')
    vocab = Vocabulary.new(gens.letter_list(args.corpus), args.vocab)

    trace('start training ...')
    model = RNNSegmentationModel.new(vocab, args.embed, args.hidden)

    for epoch in range(args.epoch):
        trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
        trained = 0

        model.init_optimizer()

        with open(args.corpus) as fp:
            for text in fp:
                word_list = text.split()
                if not word_list:
                    continue

                text = ' '.join(word_list)
                letters = ''.join(word_list)
                scores = model.train(text)
                trained += 1
                hyp = make_hyp(letters, scores)
                
                trace(trained)
                trace(text)
                trace(hyp)
                trace(' '.join('%+.1f' % x for x in scores))
                
                if trained % 100 == 0:
                    trace('  %8d' % trained)

        trace('saveing model ...')
        model.save(args.model + '.%03d' % (epoch + 1))

    trace('finished.')


def test_model(args):
    trace('loading model ...')
    model = RNNSegmentationModel.load(args.model)
    
    trace('generating output ...')

    with open(args.corpus) as fp:
        for text in fp:
            letters = ''.join(text.split())
            if not letters:
                print()
                continue
            scores = model.predict(text)
            hyp = make_hyp(letters, scores)
            print(hyp)

    trace('finished.')


def main():
    args = parse_args()

    trace('initializing ...')
    wrapper.init()

    if args.mode == 'train': train_model(args)
    elif args.mode == 'test': test_model(args)


if __name__ == '__main__':
    main()

