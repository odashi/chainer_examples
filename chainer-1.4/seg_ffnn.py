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

from util.chainer_cpu_wrapper import wrapper
#from util.chainer_gpu_wrapper import wrapper


class SegmentationModel:
    def __init__(self):
        pass

    def __make_model(self):
        self.__model = wrapper.make_model(
            w_xh = functions.EmbedID(2 * self.__n_context * len(self.__vocab), self.__n_hidden),
            w_hy = functions.Linear(self.__n_hidden, 1),
        )

    @staticmethod
    def new(vocab, n_context, n_hidden):
        self = SegmentationModel()
        self.__vocab = vocab
        self.__n_context = n_context
        self.__n_hidden = n_hidden
        self.__make_model()
        return self

    def save(self, filename):
        with ModelFile(filename, 'w') as fp:
            self.__vocab.save(fp.get_file_pointer())
            fp.write(self.__n_context)
            fp.write(self.__n_hidden)
            wrapper.begin_model_access(self.__model)
            fp.write_embed(self.__model.w_xh)
            fp.write_linear(self.__model.w_hy)
            wrapper.end_model_access(self.__model)

    @staticmethod
    def load(filename):
        self = SegmentationModel()
        with ModelFile(filename) as fp:
            self.__vocab = Vocabulary.load(fp.get_file_pointer())
            self.__n_context = int(fp.read())
            self.__n_hidden = int(fp.read())
            self.__make_model()
            wrapper.begin_model_access(self.__model)
            fp.read_embed(self.__model.w_xh)
            fp.read_linear(self.__model.w_hy)
            wrapper.end_model_access(self.__model)
        return self

    def init_optimizer(self):
        self.__opt = optimizers.AdaGrad(lr=0.01)
        self.__opt.setup(self.__model)

    def __make_input(self, is_training, text):
        c = self.__vocab.stoi
        k = self.__n_context - 1
        word_list = text.split()
        letters = [c('<s>')] * k + [c(x) for x in ''.join(word_list)] + [c('</s>')] * k
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
        letters, labels = self.__make_input(is_training, text)
        scores = []
        accum_loss = wrapper.zeros(()) if is_training else None
            
        for n in range(len(letters) - 2 * self.__n_context + 1):
            s_hu = wrapper.zeros((1, self.__n_hidden))
            
            for k in range(2 * self.__n_context):
                wid = k * len(self.__vocab) + letters[n + k]
                s_x = wrapper.make_var([wid], dtype=np.int32)
                s_hu += m.w_xh(s_x)
            
            s_hv = tanh(s_hu)
            s_y = tanh(m.w_hy(s_hv))
            scores.append(float(wrapper.get_data(s_y)))
            
            if is_training:
                s_t = wrapper.make_var([[labels[n]]])
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
    def_hidden = 100
    def_epoch = 100
    def_context = 3

    p = ArgumentParser(description='Word segmentation using feedforward neural network')

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
    model = SegmentationModel.new(vocab, args.context, args.hidden)

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
    model = SegmentationModel.load(args.model)
    
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

    trace('initializing CUDA ...')
    wrapper.init()

    if args.mode == 'train': train_model(args)
    elif args.mode == 'test': test_model(args)


if __name__ == '__main__':
    main()

