import abc

class Data(abc.ABC):
    
    @abc.abstractmethod
    def sample(self):
        pass

class Model(abc.ABC):
    
    @abc.abstractmethod
    def fit(self):
        pass
    
    @abc.abstractmethod
    def predict(self):
        pass

class ParamGen(abc.ABC):
    
    @abc.abstractmethod
    def gen(self):
        pass
    
    @abc.abstractmethod
    def get_size(self):
        pass
