# encoding = utf-8 


import tensorflow as tf


class AttnDecoder(Decoder):
    def __init__(self, isTraining, **attr, hp):
        '''
        Init class from base class
        '''
        super(AttnDecoder, self).__init__(isTraining, **attr, hp)
