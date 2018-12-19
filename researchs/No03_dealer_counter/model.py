# coding: utf-8
# chainerモデル部分

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable
from chainer import optimizers
import numpy as np


class Generator(chainer.Chain):
    def __init__(self, ):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 32, 4, 2, 1)
            self.conv2=L.Convolution2D(None, 64, 4, 2, 1)
            self.norm2=L.BatchNormalization(64)
            self.conv3=L.Convolution2D(None, 128, 4, 2, 1)
            self.norm3=L.BatchNormalization(128)
            self.conv4=L.Convolution2D(None, 256, 4, 2, 1)
            self.norm4=L.BatchNormalization(256)

            self.deconv1=L.Deconvolution2D(None, 128, 4, 2, 1)
            self.dnorm1=L.BatchNormalization(128)
            self.deconv2=L.Deconvolution2D(None, 64, 4, 2, 1)
            self.dnorm2=L.BatchNormalization(64)
            self.deconv3=L.Deconvolution2D(None, 32, 4, 2, 1)
            self.dnorm3=L.BatchNormalization(32)
            self.deconv4=L.Deconvolution2D(None, 3, 4, 2, 1)


    def __call__(self, x):
        # convolution
        h1 = F.leaky_relu(self.conv1(x))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)))
        h4 = F.leaky_relu(self.norm4(self.conv4(h3)))

        # deconvolution
        dh1 = F.leaky_relu(self.dnorm1(self.deconv1(h4)))
        dh2 = F.leaky_relu(self.dnorm2(self.deconv2(dh1)))
        dh3 = F.leaky_relu(self.dnorm3(self.deconv3(dh2)))
        #y = F.tanh(self.deconv4(dh3))
        y = self.deconv4(dh3)
        return y

class Discriminator(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 32, 4, 2, 1)
            self.conv2=L.Convolution2D(None, 64, 4, 2, 1)
            self.norm2=L.BatchNormalization(64)
            self.conv3=L.Convolution2D(None, 128, 4, 2, 1)
            self.norm3=L.BatchNormalization(128)
            self.conv4=L.Convolution2D(None, 64, 4, 2, 1)
            self.norm4=L.BatchNormalization(64)
            self.conv5=L.Convolution2D(None, 1, 4)


    def __call__(self, x):
        # convolution
        h1 = F.leaky_relu(self.conv1(x))
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)))
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)))
        h4 = F.leaky_relu(self.norm4(self.conv4(h3)))
        y = self.conv5(h4)
        return y, [h2, h3, h4]
