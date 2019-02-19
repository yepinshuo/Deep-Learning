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
from load_data import train_X, train_Y, test_X

# load data
# train_X = nd.array(np.load("./data/clean/train_X.npy"))
# train_Y = nd.array(np.load("./data/clean/train_Y.npy"))
# test_X = nd.array(np.load("./data/clean/test_X.npy"))
# print(train_X.shape, train_Y.shape)

###############################################################################
# train_data = pd.read_csv('./data/raw/kaggle_house_pred_train.csv')
# test_data = pd.read_csv('./data/raw/kaggle_house_pred_test.csv')
# all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# all_features[numeric_features] = all_features[numeric_features].apply(
#     lambda x: (x - x.mean()) / (x.std()))
# all_features = all_features.fillna(0)
# all_features = pd.get_dummies(all_features, dummy_na=True)
# n_train = train_data.shape[0]
# train_X = nd.array(all_features[:n_train].values)
# test_features = nd.array(all_features[n_train:].values)
# train_Y = nd.array(train_data.SalePrice.values).reshape((-1, 1))
###############################################################################


demo_net = nn.Sequential()
demo_net.add(nn.Dense(1))
# demo_net.initialize()

def log_rmse(net, features, labels):
    l2_loss = gloss.L2Loss()
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * l2_loss(clipped_preds.log(), labels.log()).mean())
    return rmse

demo_model = Model(model_name = "demo",
                    new_net = demo_net,
                    loss_func_ = log_rmse)

demo_model.learning_rate = 1e-4
demo_model.num_epochs = 800
demo_model.batch_size = 256



