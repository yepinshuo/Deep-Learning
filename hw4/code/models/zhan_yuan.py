from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, utils
import numpy as np
import pandas as pd
import d2l
import matplotlib.pyplot as plt
import sys
sys.path.append("./modules/Model")
sys.path.append("./modules/preprocessing")
from Model import Model

zhanyuan_net = nn.Sequential()
zhanyuan_net.add(nn.Dense(1000, activation='relu'))
zhanyuan_net.add(nn.Dropout(0.5))
zhanyuan_net.add(nn.Dense(1000, activation='relu'))
zhanyuan_net.add(nn.Dropout(0.7))
zhanyuan_net.add(nn.Dense(1000, activation='relu'))
zhanyuan_net.add(nn.Dropout(0.8))
zhanyuan_net.add(nn.Dense(1))
zhanyuan_net.initialize()

def log_rmse(net, features, labels):
    l2_loss = gloss.L2Loss()
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * l2_loss(clipped_preds.log(), labels.log()).mean())
    return rmse

ZY_model = Model(model_name = "zhanyuan",
                    new_net = zhanyuan_net,
                    loss_func_ = log_rmse)

ZY_model.learning_rate = 2e-4
ZY_model.num_epochs = 800
ZY_model.batch_size = 256
ZY_model.weight_decay = 0
