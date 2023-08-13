from pathlib import Path

# fairseq에선 @dataclass를 사용하여 config를 만듦. 
# from dataclasses import dataclass, field

## Follows torch style: https://github.com/pytorch/fairseq/blob/master/fairseq/dataclass/configs.py
class BaseConfig:
    ROOT_DIR = Path('.').resolve()
    
    print_every = 2000
    epochs = 3
    lr = 1e-5
    batch_size = 1
    loss = "CrossEntropyLoss"
    optim = "SGD"

    @classmethod
    def get_dict(cls):
        if cls is BaseConfig:
            cls_dict = {k:getattr(cls, k) for k in dir(cls) if not k.startswith('_') and not callable(getattr(cls,k))}
        else:
            cls_dict = BaseConfig.get_dict()
            cls_dict.update({k:getattr(cls, k) for k in dir(cls) if not k.startswith('_') and not callable(getattr(cls,k))})
        
        return cls_dict
    

class NewConfig(BaseConfig):
    epochs = 2
    lr = 5e-5
    optim = "Adam"
    dataset = "CIFAR10"

## tf style is different: https://github.com/tensorflow/models/blob/238922e98d/official/nlp/bert/configs.py

# from pathlib import Path

# import copy
# import json

# import six


# class BaseConfig:

#   def __init__(self,
#                 print_every=100,
#                 epochs=1,
#                 lr=1e-2,
#                 batch_size=16,
#                 loss="CrossEntropyLoss",
#                 optim="Adam"):
    
#     self.ROOT_DIR = vocab_size
    
#     self.print_every = print_every
#     self.epochs = epochs
#     self.lr = lr
#     self.batch_size = batch_size
#     self.loss = loss
#     self.optim = optim

#   @classmethod
#   def from_dict(cls, json_object):
#     """Constructs a `BaseConfig` from a Python dictionary of parameters."""
#     config = BaseConfig()
#     for (key, value) in six.iteritems(json_object):
#       config.__dict__[key] = value
#     return config

#   @classmethod
#   def from_json_file(cls, json_file, encoding='utf-8'):
#     """Constructs a `BaseConfig` from a json file of parameters."""

#     with open(json_file, 'r', encoding=encoding) as j:
#         text = j.read()
#     return cls.from_dict(json.loads(text))

#   def to_dict(self):
#     """Serializes this instance to a Python dictionary."""
#     output = copy.deepcopy(self.__dict__)
#     return output

#   def to_json_string(self):
#     """Serializes this instance to a JSON string."""
#     return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"