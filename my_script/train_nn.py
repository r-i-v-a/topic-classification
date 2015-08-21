#!/usr/bin/env python
import sys
import os
import logging
import numpy as np
import re

os.environ['THEANO_FLAGS']='nvcc.flags=-arch=sm_30,mode=FAST_RUN,device=cpu,floatX=float32'
import theano
import theano.tensor  as T

sys.path.append('/mnt/matylda4/glembek/lib/pytel-2014-12-06')
import pytel.htk      as htk
import pytel.utils    as utils
import pytel.ioutils  as ioutils

import nn_def

################################################################################
################################################################################

CWD         = os.path.dirname(os.path.realpath(__file__))
EXP_NAME    = re.sub('.py$', '', os.path.basename(__file__))
WORK_DIR    = CWD + '/' + EXP_NAME + '.dir'
CACHE_DIR   = '/mnt/scratch04/glembek/py_cache'

IN_DIR      = '/mnt/scratch01/tmp/matejkap/BEST2011/FEA_ANALYSIS_2014/BEST11__MFCC0DA120-3800-CMVN__VAD1__512G400i250lda.dir/UBM/hmms/UBM_0512G_10/iXtractor/train_iXtractor.dim400.dir/generate_ivec.v400_iter10/ascii-out'
OUT_DIR     = '/mnt/matylda4/matejkap/BEST2011/FEA_ANALYSIS_2014/BEST11__MFCC0DA120-3800-CMVN__VAD1__2048G600i250lda.dir/UBM/hmms/UBM_2048G_10/iXtractor/train_iXtractor.dim600.dir/generate_ivec.v600_iter10/ascii-out_matylda4'

TRN_SCP_FILE = '/mnt/matylda4/matejkap/BEST2011/FEA_ANALYSIS_2014/lists_BEST11/ivec.scp'
#TRN_SCP_FILE = '/homes/kazi/glembek/tmp/ivec.scp'

hidden_layer_sizes  = (1200,)
lr                  = 0.0004
batch_size          = 80
max_epochs          = 100
tolerance           = 0.00002

################################################################################
################################################################################

utils.mkdir_p(WORK_DIR)

out_nn_weights  = WORK_DIR + '/nn_weights'

# utils.mkdir_p(os.path.dirname(out_nn_weights))

logging.basicConfig(filename=WORK_DIR+'/nn_training.log', format='%(asctime)s: %(message)s', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info("Working dir set to " + WORK_DIR)

todo_list_file  = WORK_DIR + '/todo.scp'

if os.path.isfile(todo_list_file):
    TRN_SCP_FILE  = todo_list_file
    allow_missing = False
else:
    allow_missing = True


trn_list     = htk.read_scp(TRN_SCP_FILE)

in_vec_hash  = ioutils.compute_list_hash(trn_list[:,1], prefix=IN_DIR + '/',  suffix='.i.gz')
out_vec_hash = ioutils.compute_list_hash(trn_list[:,1], prefix=OUT_DIR + '/', suffix='.i.gz')

logging.info("Loading input vectors")
in_vec,in_missing   = ioutils.load_gzvectors_into_ndarray(trn_list[:,1], prefix=IN_DIR + '/',  suffix='.i.gz', allow_missing=allow_missing, dtype=np.float32)
logging.info("Loading output vectors")
out_vec,out_missing = ioutils.load_gzvectors_into_ndarray(trn_list[:,1], prefix=OUT_DIR + '/', suffix='.i.gz', allow_missing=allow_missing, dtype=np.float32)

missing      = list(set(in_missing + out_missing))

trn_list     = np.delete(trn_list, missing, axis=0)
in_vec       = np.delete(in_vec,  missing, axis=0)
out_vec      = np.delete(out_vec, missing, axis=0)

# make every 10th sample as CV
cv_idx       = np.arange(0, in_vec.shape[0], 10)
in_cv_vec    = in_vec.take(cv_idx, axis=0)
out_cv_vec   = out_vec.take(cv_idx, axis=0)

in_vec       = np.delete(in_vec,  cv_idx, axis=0)
out_vec      = np.delete(out_vec, cv_idx, axis=0)

in_vec_size  = in_vec.shape[1]
out_vec_size = out_vec.shape[1]

#hidden_layer_sizes=(1500, 1500, 1500)

# Calculate mean and std at the NN input
logging.info("Estimating mean and std at the NN input")

input_mean = np.mean(in_vec, axis=0).astype("float32")
input_std  = np.std(in_vec, axis=0).astype("float32")

# Neural network definition
logging.info("Creating and initializing NN")
np.random.seed(42)
X_, Y_, params_ = nn_def.create_theano_regression_nn(
    nn_def.init_regression_params(input_mean, input_std, hidden_layer_sizes, 
    out_vec_size))

lr_               = T.scalar()
T_                = T.matrix("T")
#cost_             = T.sum((Y_ - T_) ** 2, axis=1).sum()/Y_.shape[0]
cost_             = T.sum((Y_ - T_) ** 2, axis=1).sum()
params_to_update_ = [p for p in params_ if p.name[0] in "Wb"]
grads_            = T.grad(cost_, params_to_update_)

train = theano.function(
    inputs=[X_, T_, lr_],
    outputs=cost_,
    updates=[(p, p - lr_ * g) for p, g in zip(params_to_update_, grads_)])

mse = theano.function(inputs=[X_, T_], outputs=cost_)

last_cv_error   = np.inf
lr_decay_factor = 1

logging.info("Training model: %s\nlearning rate: %f\nmini batch size: %d\nmax iters: %d tolerance: %f" % (
             '->'.join(map(str, (len(input_mean),)+hidden_layer_sizes+(out_vec_size,))), lr, batch_size, max_epochs, tolerance))

for kk in range(1,max_epochs):
    lr   *= lr_decay_factor
    error = 0.0
    n     = 0

    logging.info("Training epoch: %d, learning rate: %f", kk, lr)

    shuffle  = np.random.permutation(in_vec.shape[0])

    in_vec  = in_vec.take(shuffle,  axis = 0)
    out_vec = out_vec.take(shuffle, axis = 0) #faster than fea[shuffle]

    nsplits  = len(in_vec)/batch_size

    for jj, (X, Y) in enumerate(zip(np.array_split(in_vec, nsplits), 
                     np.array_split(out_vec, nsplits))):
        err     = train(X.astype(np.float32), Y.astype(np.float32), lr)
        error  += err
        n      += len(X)

    logging.info("%d | %f", n, error / n)

    logging.info("Evaluating on CV")

    error = 0.0
    n     = 0

    for jj, (X, Y) in enumerate(zip(np.array_split(in_cv_vec, nsplits), np.array_split(out_cv_vec, nsplits))):
        err     = mse(X.astype(np.float32), Y.astype(np.float32))
        error  += err
        n      += X.shape[0]

    logging.info("%d | %f", in_cv_vec.shape[0], error / n)

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

