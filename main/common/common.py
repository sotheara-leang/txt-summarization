import logging
import torch as t

from main.conf.configuration import Configuration

#
conf = Configuration()

#

def cuda(tensor):
    if t.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def getLogger(self):
    return logging.getLogger(self.__class__.__name__)
