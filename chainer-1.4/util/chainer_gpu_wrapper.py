import numpy
import chainer

class wrapper:
    @staticmethod
    def init():
        chainer.cuda.init()

    @staticmethod
    def make_var(array, dtype=numpy.float32):
        return chainer.Variable(chainer.cuda.to_gpu(numpy.array(array, dtype=dtype)))

    @staticmethod
    def get_data(variable):
        return chainer.cuda.to_cpu(variable.data)

    @staticmethod
    def zeros(shape, dtype=numpy.float32):
        return chainer.Variable(chainer.cuda.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(shape, dtype=numpy.float32):
        return chainer.Variable(chainer.cuda.ones(shape, dtype=dtype))

    @staticmethod
    def make_model(**kwargs):
        return chainer.FunctionSet(**kwargs).to_gpu()        
 
    @staticmethod
    def begin_model_access(model):
        model.to_cpu()

    @staticmethod
    def end_model_access(model):
        model.to_gpu()

