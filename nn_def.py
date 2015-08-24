#!THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 ipython
import theano, theano.tensor as T
import pytel.features as features
import numpy as np

left_ctx = right_ctx = 15

dct_basis = 16
hamming_dct = (features.dct_basis(dct_basis, left_ctx+right_ctx+1)*np.hamming(left_ctx+right_ctx+1)).T.astype("float32")

def preprocess_nn_input(X):
    X = features.framing(X, left_ctx+1+right_ctx).transpose(0,2,1)
    return np.dot(X.reshape(-1,hamming_dct.shape[0]), hamming_dct).reshape(X.shape[0], -1)

def create_theano_nn(param_dict):
    X_              = T.matrix("X")
    mean_           = theano.shared(param_dict['input_mean'], name = 'input_mean')
    std_            = theano.shared(param_dict['input_std'], name  = 'input_std')
    Y_              = (X_ - mean_) / std_
    params_         = [mean_, std_]
    n_hidden_layers = len(param_dict.keys())/2-2
  
    for ii, f in enumerate([T.nnet.sigmoid]*n_hidden_layers+[T.nnet.softmax]):
        W_        = theano.shared(param_dict['W'+str(ii+1)], name = 'W'+str(ii+1))
        b_        = theano.shared(param_dict['b'+str(ii+1)], name = 'b'+str(ii+1))
        if f == None: Y_        =   Y_.dot(W_) + b_
        else:         Y_        = f(Y_.dot(W_) + b_)

        params_  += [W_, b_]
  
    return X_, Y_, params_


def create_theano_regression_nn(param_dict):
    X_              = T.matrix("X")
    mean_           = theano.shared(param_dict['input_mean'], name = 'input_mean')
    std_            = theano.shared(param_dict['input_std'], name  = 'input_std')
    Y_              = (X_ - mean_) / std_
    params_         = [mean_, std_]
    n_hidden_layers = len(param_dict.keys())/2-2
  
    for ii, f in enumerate([T.nnet.sigmoid]*n_hidden_layers+[None]):
        W_        = theano.shared(param_dict['W'+str(ii+1)], name = 'W'+str(ii+1))
        b_        = theano.shared(param_dict['b'+str(ii+1)], name = 'b'+str(ii+1))
        if f == None: Y_        =   Y_.dot(W_) + b_
        else:         Y_        = f(Y_.dot(W_) + b_)

        params_  += [W_, b_]
  
    return X_, Y_, params_

def init_params(input_mean, input_std, hidden_layer_sizes, nclasses):
    sizes = (len(input_mean),)+tuple(hidden_layer_sizes)+(nclasses,)
    params_dict = {"input_mean": input_mean.astype(T.config.floatX), 
                   "input_std":  input_std.astype(T.config.floatX)}
    for ii in range(1,len(sizes)):   params_dict['W'+str(ii)] = np.random.randn(sizes[ii-1],sizes[ii]).astype(T.config.floatX)*0.1
    for ii in range(1,len(sizes)-1): params_dict['b'+str(ii)] = np.random.random(           sizes[ii]).astype(T.config.floatX)/5.0-4.1
    params_dict['b'+str(len(sizes)-1)] = np.zeros(sizes[len(sizes)-1]).astype(T.config.floatX)
    return params_dict

def init_regression_params(input_mean, input_std, hidden_layer_sizes, out_vec_size):
    sizes = (len(input_mean),)+tuple(hidden_layer_sizes)+(out_vec_size,)
    params_dict = {"input_mean": input_mean.astype(T.config.floatX), 
                   "input_std":  input_std.astype(T.config.floatX)}
    for ii in range(1,len(sizes)):   params_dict['W'+str(ii)] = np.random.randn(sizes[ii-1],sizes[ii]).astype(T.config.floatX)*0.1
    for ii in range(1,len(sizes)-1): params_dict['b'+str(ii)] = np.random.random(           sizes[ii]).astype(T.config.floatX)/5.0-4.1
    params_dict['b'+str(len(sizes)-1)] = np.zeros(sizes[len(sizes)-1]).astype(T.config.floatX)
    return params_dict

def get_params(params_):
    return {p.name: p.get_value() for p in params_}

def set_params(params_, param_dict):
    for p_ in params_: p_.set_value(param_dict[p_.name])
