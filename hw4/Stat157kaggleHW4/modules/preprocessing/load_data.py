import numpy as np
from mxnet import nd
import mxnet as mx

train_X, train_Y, test_X = None, None, None

try:
    train_X = nd.array(np.load("./data/clean/train_X.npy"), ctx=mx.gpu())
    train_Y = nd.array(np.load("./data/clean/train_Y.npy"), ctx=mx.gpu())
    test_X = nd.array(np.load("./data/clean/test_X.npy"), ctx=mx.gpu())
except:
    print("Failed to load into GPU; loading into memory")
    train_X = nd.array(np.load("./data/clean/train_X.npy"))
    train_Y = nd.array(np.load("./data/clean/train_Y.npy"))
    test_X = nd.array(np.load("./data/clean/test_X.npy"))
