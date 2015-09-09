from .functions import vtos, stov

class ModelFile:
    def __init__(self, filename, mode='r'):
        self.__fp = open(filename, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__fp.close()
        return False

    def write(self, x):
        print(x, file=self.__fp)

    def __write_vector(self, x):
        self.write(vtos(x))

    def __write_matrix(self, x):
        for row in x:
            self.__write_vector(row)
    
    def read(self):
        return next(self.__fp).strip()

    def __read_vector(self, x, tp):
        data = stov(self.read(), tp)
        for i in range(len(data)):
            x[i] = data[i]

    def __read_matrix(self, x, tp):
        for row in x:
            self.__read_vector(row, tp)

    def write_embed(self, f):
        self.__write_matrix(f.W)

    def write_linear(self, f):
        self.__write_matrix(f.W)
        self.__write_vector(f.b)

    def read_embed(self, f):
        self.__read_matrix(f.W, float)

    def read_linear(self, f):
        self.__read_matrix(f.W, float)
        self.__read_vector(f.b, float)

    def get_file_pointer(self):
        return self.__fp

