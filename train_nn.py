#!/usr/bin/env python
import sys
import os
import logging
import numpy as np
import re

os.environ['THEANO_FLAGS']='nvcc.flags=-arch=sm_30,mode=FAST_RUN,device=cpu,floatX=float32'
import theano
import theano.tensor  as T

sys.path.append('/homes/kazi/burget/pytel/')
import pytel.htk      as htk
import pytel.utils    as utils
import pytel.ioutils  as ioutils

import nn_def

################################################################################
################################################################################

CWD         = os.path.dirname(os.path.realpath(__file__))
EXP_NAME    = re.sub('.py$', '', os.path.basename(__file__))
WORK_DIR    = CWD + '/' + EXP_NAME + '.dir'

hidden_layer_sizes  = (2000, 2000, 2000)
lr                  = 0.008
batch_size          = 80
max_epochs          = 100
tolerance           = 0.003
nclasses            = 40

################################################################################
################################################################################

utils.mkdir_p(WORK_DIR)

out_nn_weights  = WORK_DIR + '/nn_weights'

# utils.mkdir_p(os.path.dirname(out_nn_weights))

# prints info to screen while running
logging.basicConfig(filename=WORK_DIR+'/nn_training.log', format='%(asctime)s: %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info("Working dir set to " + WORK_DIR)

logging.info("Loading input vectors")
in_vec = np.loadtxt('../data/400_a_train')[:,1:]

logging.info("Loading output vectors")
out_vec = np.loadtxt('../data/400_a_train')[:,0] - 1

# make every 10th sample as CV
cv_idx       = np.arange(0, in_vec.shape[0], 10)
in_cv_vec    = in_vec.take(cv_idx, axis=0)
out_cv_vec   = out_vec.take(cv_idx, axis=0)

in_vec       = np.delete(in_vec,  cv_idx, axis=0)
out_vec      = np.delete(out_vec, cv_idx, axis=0)

in_vec_size  = in_vec.shape[1]
out_vec_size = out_vec.shape[0]

# Calculate mean and std at the NN input
logging.info("Estimating mean and std at the NN input")

input_mean = np.mean(in_vec, axis=0).astype("float32")
input_std  = np.std(in_vec, axis=0).astype("float32")

# Neural network definition
logging.info("Creating and initializing NN")
np.random.seed(42)
X_, Y_, params_ = nn_def.create_theano_nn(
    nn_def.init_params(input_mean, input_std, hidden_layer_sizes, 
    nclasses))

lr_               = T.scalar()
T_                = T.ivector("T")
cost_             = T.nnet.categorical_crossentropy(Y_, T_).sum()
acc_              = T.eq(T.argmax(Y_, axis=1), T_).sum()
params_to_update_ = [p for p in params_ if p.name[0] in "Wb"]
grads_            = T.grad(cost_, params_to_update_)

train = theano.function(
    inputs=[X_, T_, lr_],
    outputs=[cost_, acc_],
    updates=[(p, p - lr_ * g) for p, g in zip(params_to_update_, grads_)])

xentropy = theano.function(inputs=[X_, T_], outputs=[cost_, acc_])

last_cv_error   = np.inf
lr_decay_factor = 1

logging.info("Training model: %s\nlearning rate: %f\nmini batch size: %d\nmax iters: %d tolerance: %f" % (
             '->'.join(map(str, (len(input_mean),)+hidden_layer_sizes+(nclasses,))), lr, batch_size, max_epochs, tolerance))

for kk in range(1,max_epochs):
    lr        *= lr_decay_factor
    error      = 0.0
    accuracy   = 0.0
    n          = 0

    logging.info("Training epoch: %d, learning rate: %f", kk, lr)

    shuffle  = np.random.permutation(in_vec.shape[0])

    in_vec  = in_vec.take(shuffle,  axis = 0)
    out_vec = out_vec.take(shuffle, axis = 0) #faster than fea[shuffle]

    nsplits  = len(in_vec)/batch_size

    for jj, (X, t) in enumerate(zip(np.array_split(in_vec, nsplits), 
                     np.array_split(out_vec, nsplits))):
        err, acc     = train(X.astype(np.float32), t.astype(np.int16), lr)
        error       += err
        accuracy    += acc
        n           += len(X)

    logging.info("%d | %f | %f", n, error / n, accuracy / n)

    logging.info("Evaluating on CV")

    error    = 0.0
    accuracy = 0.0
    n        = 0

    for jj, (X, t) in enumerate(zip(np.array_split(in_cv_vec, nsplits), np.array_split(out_cv_vec, nsplits))):
        err, acc  = xentropy(X.astype(np.float32), t.astype(np.int16))
        error    += err
        accuracy += acc
        n        += X.shape[0]

    logging.info("%d | %f | %f", n, error / n, accuracy / n)

    if last_cv_error <= error: # load previous weights if error increases
        nn_def.set_params(params_, last_params)
        error = last_cv_error

    if (last_cv_error-error)/np.abs([last_cv_error, error]).max() <= tolerance: # start halving learning rate or terminate the training 
        if lr_decay_factor < 1: break
        lr_decay_factor = 0.5

    last_cv_error = error
    last_params = nn_def.get_params(params_)

np.savez(out_nn_weights+'_final', **nn_def.get_params(params_))
#xx

