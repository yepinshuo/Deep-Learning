import d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, utils
import numpy as np
import pandas as pd
import sys
sys.path.append("./modules/Model")
sys.path.append("./modules/preprocessing")
sys.path.append("./models")
from Model import Model
from load_data import train_X, train_Y, test_X

# print(train_X.shape)
# print(test_X.shape)

# Change to your own model
from demo import demo_model
from zhan_yuan import ZY_model

# demo_model.train(train_features=train_X, train_labels=train_Y, print_iter=True)
# demo_model.export_predict(test_X)

noised_train_X = train_X + nd.random.normal(0, 0.01, shape=train_X.shape)
noised_train_Y = train_Y + nd.random.normal(0, 0.01, shape=train_Y.shape)

aug_train_X = nd.concat(train_X, noised_train_X, dim=0)
aug_train_Y = nd.concat(train_Y, noised_train_Y, dim=0)

# ZY_model.train_k_fold_cv(train_features = aug_train_X,
#                         train_labels = aug_train_Y,
#                         force_reinit=False)

ZY_model.train(aug_train_X, aug_train_Y)
ZY_model.export_predict(test_X, path_="./submission_PY.csv")