from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, utils
import mxnet as mx
import numpy as np
import pandas as pd
import d2l
import matplotlib.pyplot as plt

class Model:

    # This is the class, an instance of which represents
    # a neural network
    # a trainer
    # and a predictor

    # At the instantiation of a Model instance, the following things
    # but be specified:
    #   model_name
    #       str, a name for the model
    #   new_net:
    #       the Gluon NN object that represents the computational graph
    #   loss_func:
    #       The loss function that takes a net object, a feature matrix and a
    #       response matrix, and return an NDArray object with identical
    #       shape as the response matrix.
    def __init__(self, model_name, new_net, loss_func_):
        self.name = model_name
        self.net = new_net
        self.loss_func = loss_func_

        # A number of default instance attributes will be created, too
        self.learning_rate = 1
        self.num_epochs = 0
        self.weight_decay = 0
        self.batch_size = 10
        self.train_losses = []
        self.val_losses = []
        self.train_losses_cv = []
        self.val_losses_cv = []

        # Attempt to initialize on GPU. If failed, initialize on CPU
#        try:
#            self.net.initialize(ctx=mx.gpu())
#        except:
#            self.net.initialize()
            
#    def force_reinit(self):
        # Forcefully re-initialize the parameters
        # Attempt to initialize on GPU. If failed, initialize on CPU
#        try:
#            self.net.initialize(force_reinit=True, ctx=mx.gpu())
#        except:
#            self.net.initialize(force_reinit=True, cts=mx.cpu())

    def train(self, train_features, train_labels,
        test_features=None, test_labels=None,
        shuffle_=True, print_iter=True):
        train_losses, test_losses = [], []
        train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), self.batch_size, shuffle=shuffle_)

        trainer = gluon.Trainer(self.net.collect_params(),
            'adam',
            {
            'learning_rate': self.learning_rate,
                'wd': self.weight_decay
            })

        for epoch in range(self.num_epochs):
            for X, y in train_iter:
                with autograd.record():
                    l = self.loss_func(self.net, X, y)
                l.backward()
                trainer.step(self.batch_size)
            train_losses.append(self.loss_func(self.net,
                                            train_features,
                                            train_labels).asscalar())
            if print_iter:
                print("Epoch: " + str(epoch + 1))
                print("\t Training_Loss: " + str(train_losses[-1]))
            if test_labels is not None:
                test_losses.append(self.loss_func(self.net,
                                            test_features,
                                            test_labels).asscalar())
        self.train_losses = train_losses
        self.test_losses = test_losses
        return train_losses, test_losses

    def get_k_fold_data(self, k, i, X, y):
        assert k > 1
        fold_size = X.shape[0] // k
        X_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = nd.concat(X_train, X_part, dim=0)
                y_train = nd.concat(y_train, y_part, dim=0)
        return X_train, y_train, X_valid, y_valid

    def train_k_fold_cv(self, train_features, train_labels, k=5, force_reinit=True):
        self.train_losses_cv = []
        self.val_losses_cv = []
        for i in range(k):
            #if force_reinit:
                #self.force_reinit()
            train_X, train_Y, val_X, val_Y = self.get_k_fold_data(k, i, train_features, train_labels)
            train_losses, val_losses = self.train(train_features=train_X,
                                                train_labels=train_Y,
                                                test_features=val_X,
                                                test_labels=val_Y,
                                                print_iter=False)
            print('fold %d, train rmse: %f, valid rmse: %f' % (
            i, train_losses[-1], val_losses[-1]))
            self.train_losses_cv.append(train_losses)
            self.val_losses_cv.append(val_losses)

    def predict(self, test_features, shift=0, scale=1):
        return self.net(test_features) * scale + shift

    def export_predict(self, test_features, path_="./submission.csv", shift=0, scale=1):
        preds = self.predict(test_features, shift, scale).asnumpy()
        Ids = list(range(1461, 2919+1))
        submission = pd.DataFrame()
        submission["Id"] = Ids
        submission["SalePrice"] = preds
        submission.to_csv(path_, index=False)

