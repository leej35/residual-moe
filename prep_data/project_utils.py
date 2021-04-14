# -*- coding: utf-8 -*-
"""
07 Oct 2017
A collection of utility packages
"""

# Import statements
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import time
import logging
from datetime import datetime
import os
import dill
# import cPickle as pickle
import pickle


def save_obj(obj, filename):
    with open(filename, 'wb') as output:
        dill.dump(obj, output)


def load_obj(filename):
    with open(filename, 'rb') as inputf:
        return dill.load(inputf)

def cp_save_obj(obj, filename):
    os.system("mkdir -p {}".format(os.path.split(filename)[0]))
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def cp_load_obj(filename):
    with open(filename, 'rb') as inputf:
        return pickle.load(inputf)

class Timing(object):
    """A context manager that prints the execution time of the block it manages"""

    def __init__(self, message, file=sys.stdout, logger=None, one_line=True):
        self.message = message
        if logger is not None:
            self.default_logger = False
            self.one_line = False
            self.logger = logger
        else:
            self.default_logger = True
            self.one_line = one_line
            self.logger = None
        self.file = file

    def _log(self, message, newline=True):
        if self.default_logger:
            print(message, end='\n' if newline else '', file=self.file)
            try:
                self.file.flush()
            except:
                pass
        else:
            self.logger.info(message)

    def __enter__(self):
        self.start = time.time()
        self._log(self.message, not self.one_line)

    def __exit__(self, exc_type, exc_value, traceback):
        self._log('{}Done in {:.3f}s'.format('' if self.one_line else self.message, time.time()-self.start))


def main():
    pass


if __name__ == '__main__':
    main()

