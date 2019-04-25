import logging
import torch as t

from main.conf.configuration import Configuration

#
conf = Configuration()

#
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


def cuda(tensor, device_=None):
    return tensor.to(device_ if device_ is not None else device)


def getLogger(self):
    return logging.getLogger(self.__class__.__name__)
