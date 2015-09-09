import numpy
import chainer

class wrapper:
    @staticmethod
    def init():
        pass

    @staticmethod
    def make_var(array, dtype=numpy.float32):
        return chainer.Variable(numpy.array(array, dtype=dtype))

    @staticmethod
    def get_data(variable):
        return variable.data

    @staticmethod
    def zeros(shape, dtype=numpy.float32):
        return chainer.Variable(numpy.zeros(shape, dtype=dtype))

    @staticmethod
    def ones(shape, dtype=numpy.float32):
        return chainer.Variable(numpy.ones(shape, dtype=dtype))

    @staticmethod
    def make_model(**kwargs):
        return chainer.FunctionSet(**kwargs)

    @staticmethod
    def begin_model_access(model):
        pass

    @staticmethod
    def end_model_access(model):
        pass
 
