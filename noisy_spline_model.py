"""
This script create neural networks which are 1-input, 1-output, and K layers
of N relu nodes.  The idea is to create a network which models a high-frequency
sawtooth and a low-frequency component.  Currently this is done in a somewhat
wasteful manner, where the left hand block of K x N_saw is used to construct a
sawtooth with about (N_saw-1)^k knots, and the right hand block of K x N_rand
is used to construct a random spline with N_rand knots (lots of nodes here
are unused).

When run by itself, it demonstrates the observation from the paper
"The energy landscape of a simple neural network" that it is hard to
fit a 5x5 network to the output of a moderately complicated 5x5 network.
"""
import sys
import os
import numpy as np
import math
import linear_spline as ls


def build_model(num_features, layer_data, loss='mse', optimizer='adadelta'):
    import warnings
    warnings.simplefilter('ignore')
    import keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.regularizers import l1, l2
    from keras.layers.normalization import BatchNormalization
    from keras.layers.noise import GaussianNoise
    import keras.backend as K
    model = Sequential()
    for i,ld in enumerate(layer_data):
        args = ({} if i > 0 else {'input_shape':(num_features,)})
        if ld[0] == 'dense':
            if ld[2] == 'relusq':
                model.add( Dense(ld[1], **args) )
                model.add(Activation(K.relu))
                model.add(Activation(K.square))
            else:
                model.add( Dense(ld[1], activation=ld[2], **args) )
        elif ld[0] == 'batchnorm':
            model.add( BatchNormalization(**args) )
        elif ld[0] == 'dropout':
            model.add( Dropout(ld[1], **args) )
        elif ld[0] == 'noise':
            model.add( GaussianNoise(ld[1], **args) )
        else:
            print("Unknown input")
    model.compile(loss=loss,optimizer=optimizer)
    return model

def build_model_from_weights(W, activation=None):
   """Assumes you want all relu except the last layer is linear"""
   arch = []
   num_features = W[0].shape[0]
   for i in range(0, len(W)-2, 2):
      arch.append( ('dense', W[i].shape[1], ('relu' if activation is None else activation) ) )
   arch.append( ('dense', W[-1].shape[0], 'linear') )
   ans = build_model(num_features, arch)
   ans.set_weights(W)
   return ans


def plot_model(m, constrain=(-1,1), colors=None):
    import matplotlib.pyplot as plt

    if type(m)!=list or ('shape' in dir(m[0]) and len(m[0].shape)==2) :
        m = [m]
   
    A = 'relu'
    m = [ ( x if 'summary' in dir(x) else build_model_from_weights(x, A) ) for x in m]
    f = [ls.LinearSpline.from_nn_np(M) for M in m]
   
    #print f.knots, f.heights, f.inf_slopes
    #print "There are ", len(f.knots), "knots"
    for j,F in enumerate(f):
        plottable_knots = F.knots
        plottable_heights = F.heights
        if constrain is not None:
            inds = [i for i in range(len(plottable_knots)) if constrain[0] <= plottable_knots[i] and plottable_knots[i] <= constrain[1]]
            plottable_knots = [plottable_knots[i] for i in inds]
            plottable_heights = [plottable_heights[i] for i in inds]
        plottable_knots = [constrain[0]] + plottable_knots + [constrain[1]]
        plottable_heights = [F.evaluate(constrain[0])] + plottable_heights + [F.evaluate(constrain[1])]
        if colors is not None:
            plt.plot(plottable_knots, plottable_heights, color=colors[j])
        else:
            plt.plot(plottable_knots, plottable_heights)
    plt.show()

def create_sawtooth(N, K, bias=0, amp=None):
    """Create a 1-input, 1-output, K hidden layers of N nodes network which
    produces a sawtooth wave.  It has about (N-1)^k knots.  amp is the amplitude"""

    A = 'relu'

    arch = K*[ ('dense', N, A) ]
    arch.append( ('dense', 1, 'linear') )
    m = build_model(1, arch, loss='mse')  # this is an easy way to get a bunch of numpy arrays
    W = m.get_weights()

    #Create the first layer of equally spaced knots
    W[0] = np.ones( W[0].shape )
    W[1] = np.arange( 1, -1, -2.0/W[1].shape[0] )
    #Note the height reached by each is 2.0/W[1].shape[0]
    peak_height = 2.0/W[1].shape[0]
    #Now there are N-1 straight segments in wiggles
    
    for i in range(2, len(W)-2, 2):
        #Create the next layer in which we add up all of the previous layer with alternating signs
        W[i] = np.ones( W[i].shape )
        W[i][1::2,:] = -2
        W[i][2::2,:] = 2
        W[i+1] = np.arange( 0, -peak_height, -peak_height/W[i+1].shape[0] )
        peak_height = peak_height/W[i+1].shape[0]
        #We just added N-1 knots in every straight segment, so there are
        #(previous number)*(N-1)
   
    #The final layer just adds everything with the same alternating
    W[-2] = np.ones( W[-2].shape )
    W[-2][1::2,:] = -2
    W[-2][2::2,:] = 2
    if amp is not None:
        W[-2] *= (2*amp/peak_height)
        W[-1] = np.array([bias-amp])
    else:
       W[-1] = np.array([bias-0.5*peak_height])
    #We did K layers with cutting, so we have created (N-1)^K straight segments

    m.set_weights(W)
    return m
    


def create_spline_model(N, K, ystd=1):
   """This essentially ignores K and makes a spline with N knots; it requires K>=1"""
   knots = np.arange(-1,1,2.0/N) #np.sort(np.random.uniform(-1, 1, size=N))
   heights = np.random.normal(0, ystd, size=N)
   heights = heights - heights[0] #it starts at 0 because why not
   knots.resize(N+1)
   heights.resize(N+1)
   knots[-1] = 1
   heights[-1] = 0 #ends at 0
   #print zip(knots, heights)
   
   biases = [-x for x in knots[:-1]]
   cur_slope = 0
   weights = []
   for i in range(1, len(heights)):
      next_slope = (heights[i] - heights[i-1]) / (knots[i] - knots[i-1])
      weights.append( -cur_slope + next_slope )
      cur_slope = next_slope
   arch = K*[ ('dense', N, 'relu') ]
   arch.append( ('dense', 1, 'linear') )
   m = build_model(1, arch, loss='mse')
   W = m.get_weights()
   #print W
   W[0] = np.ones(shape=W[0].shape)
   W[1] = np.array(biases)
   for i in range(2,len(W)-2,2):
      W[i] = np.identity(W[i].shape[0])
   W[-2] = np.array(weights).reshape( (-1,1) )
   #print W
   m.set_weights(W)
   return m, ls.LinearSpline(knots, heights, [0,0])



def concat_horiz(a,b):
   if len(a.shape) == 1:
      return np.concatenate( (a,b) )
   else:
      return np.concatenate( (a,b), axis=1 )


def create_noised_model(N_saw, N_rand, K, saw_amp=None, rand_std=1):
   """Create a noised model, which is the combination of a sawtooth and a random
   spline"""
   
   m_saw = create_sawtooth(N_saw, K, amp=saw_amp)
   m_rand, spline_rand = create_spline_model(N_rand, K, ystd=rand_std)
   
   W_saw = m_saw.get_weights()
   W_rand = m_rand.get_weights()
   
   #print "Saw:"
   #print W_saw
   #print "Rand:"
   #print W_rand

   #pad out the weights so that they can be concatenated
   for i in range(2, len(W_saw)-2, 2):
      W_saw[i] = np.concatenate( (W_saw[i], np.zeros( (N_rand, N_saw) ) ), axis=0 )
   for i in range(2, len(W_rand)-2, 2):
      W_rand[i] = np.concatenate( (np.zeros((N_saw, N_rand)), W_rand[i] ), axis=0 )

   #print "Saw:"
   #print W_saw
   #print "Rand:"
   #print W_rand

   #Concatenate so it's just two parallel networks
   #combined = [ np.concatenate( (W_saw[i], W_rand[i]), axis=1) for i in range(len(W_saw)-2) ]  # Doesn't work with mixed shapes?
   combined = [concat_horiz( W_saw[i], W_rand[i] ) for i in range(len(W_saw)-2)]
   #The final node needs to just add them up
   combined.extend( [ np.concatenate( (W_saw[-2], W_rand[-2]), axis=0 ), np.array([0]) ] )
   
   #print combined
   
   #Now build a model of the correct shape

   arch = K*[ ('dense', N_saw + N_rand, 'relu') ]
   arch.append( ('dense', 1, 'linear') )
   m = build_model(1, arch, loss='mse')
   m.set_weights(combined)
   
   return m


def create_random(N, K, random_bias=False):
   """Create a random model, which is initialized and then optionally has the layer 1
   biases set randomly"""
   arch = K*[ ('dense', N, 'relu') ]
   arch.append( ('dense', 1, 'linear') )
   m = build_model(1, arch, loss='mse')
   if random_bias:
      W = m.get_weights()
      W[1] = np.random.normal( 0, 1, size=W[1].shape)
      m.set_weights(W)
   return m


def fit_another_model(m,
                      starting_weights=None,
                      nb_batches=1000,
                      batch_size=1024,
                      subdivide=10,
                      ee_patience=None,
                      snapshot_interval=None,
                      optimizer='adadelta',
                      KN=None,
                      activation='relu'):
   """If batch_size==None, then it uses the knots (subdivided into 5) of m as training points"""
   W = m.get_weights()
   if KN is None:
      K = (len(W)//2)-1
      N = W[1].shape[0]
   else:
      K, N = KN
   arch = K*[ ('dense', N, activation) ]
   arch.append( ('dense', 1, 'linear') )
   m2 = build_model(1, arch, loss='mse', optimizer=optimizer)
   if starting_weights is not None:
      m2.set_weights(starting_weights)
   loss_history = []
   best_weights = None
   best_loss = None
   best_time = None
   snapshots = []
   interpolate_params = np.arange(1.0, 0.0, -1.0/subdivide)
   if batch_size is None:
      ms = ls.LinearSpline.from_nn_np(m)
      m_knots_subdiv = []
      for i in range(len(ms.knots)-1):
         m_knots_subdiv.extend( [ t*ms.knots[i] + (1-t)*ms.knots[i+1] for t in interpolate_params ] )
      m_knots_subdiv = np.array([x for x in m_knots_subdiv if -1 <= x and x <= 1])
      m_heights_subdiv = np.array([ms.evaluate(x) for x in m_knots_subdiv])
      m_knots_subdiv = m_knots_subdiv.reshape( (-1, 1) )
      m_heights_subdiv = m_heights_subdiv.reshape( (-1, 1) )
   for i in range(nb_batches):
      if batch_size is None:
         X = m_knots_subdiv
         y = m_heights_subdiv
      else:
         X = np.random.uniform( -1, 1, size=(batch_size, 1) )
         y = m.predict(X)
      loss = m2.train_on_batch(X, y)
      loss_history.append(loss)
      if snapshot_interval is not None and i%snapshot_interval == 0:
         snapshots.append(m2.get_weights())
      if ee_patience is not None:
         if best_loss is None or loss < best_loss:
            best_weights = ([x.copy() for x in m2W] if basic else m2.get_weights())
            best_loss = loss
            best_time = i
         elif best_time is not None and i-best_time > ee_patience:
            m2.set_weights(best_weights)
            m2W = best_weights
            break
   ans = {'loss_history':loss_history, 'nb_batches':best_time, 'weights':m2.get_weights(), 'snapshots':snapshots}
   return ans


if __name__ == '__main__':
  M1 = create_noised_model(2, 3, 5, 0.7, 1)
  M = [fit_another_model(M1, nb_batches=50000, batch_size=None) for i in range(3)]
  plot_model( [M1] + [x['weights'] for x in M] )















