"""
A linear spline class, especially intended to model the output
of one-input, one-output relu neural networks.  It is also
useful for ROC curves

note: in certain places, this code checks for *equality of floats*;
it should work in the way that it's doing it
"""

import numpy as np
import os


def resample_min_gaps(L, min_gap):
  """Return L subsampled so that no gap is below min_gap.  This assumes
  that L is sorted"""
  ans = [L[0]]
  for x in L[1:]:
    if x-ans[-1] > min_gap:
      ans.append(x)
  return ans


# a linear spline is given by a list of knot x-values, a list of knot y-values,
# and a pair of negative and positive infinity slopes
class LinearSpline(object):
  
  @staticmethod
  def mean_pm_stderr(splines, just_std=False, min_gap=None):
    """Returns three splines which track the mean and 2*stderr 2*(stdev/sqrt(n)) of
    the splines, interpolated between all knots on all splines.  The return
    value is (mean, mean-stderr, mean+stderr).  If just_std, it's +/- stdev"""
    m,s = LinearSpline.mean_and_stdev(splines, min_gap=min_gap)
    if not just_std:
      n = np.sqrt(float(len(splines)))
      se = 2.0*np.array(s.heights)/n
    else:
      se = np.array(s.heights)
    return (m,
            LinearSpline(m.knots, np.array(m.heights) - se, [0,0]),
            LinearSpline(m.knots, np.array(m.heights) + se, [0,0]))
  
  @staticmethod
  def mean_and_stdev(splines, min_gap=None):
    """Returns two splines which track the mean and stdev of
    the splines, interpolated between all knots on all splines.  The return
    value is (mean, std), where std is spline tracking the std."""
    all_knots = sorted(list(set([x for s in splines for x in s.knots])))
    if min_gap is not None:
      all_knots = resample_min_gaps(all_knots, min_gap)
    refined = [s.refined(all_knots) for s in splines]
    means = np.array( [ np.mean([r.heights[j] for r in refined]) for j in range(len(all_knots))] )
    meanslopes = [np.mean([r.inf_slopes[0] for r in refined]), np.mean([r.inf_slopes[1] for r in refined])]
    stdevs = np.array( [ np.std([r.heights[j] for r in refined]) for j in range(len(all_knots))] )
    return ( LinearSpline(all_knots, means, meanslopes),
             LinearSpline(all_knots, stdevs, meanslopes) )
  
  @staticmethod
  def affine_combination(weights, splines, bias):
    """Return a spline which is the affine combination of the splines given, so 
    (sum_i weight_i*spline_i) + bias, where bias is a scalar"""
    all_knots = sorted(list(set([x for s in splines for x in s.knots])))
    refined = [s.refined(all_knots) for s in splines]
    new_heights = [ sum([weights[i]*r.heights[j] for i,r in enumerate(refined)]) + bias for j in range(len(all_knots)) ]
    new_slopes = [ sum([weights[i]*r.inf_slopes[0] for i,r in enumerate(refined)]),
                   sum([weights[i]*r.inf_slopes[1] for i,r in enumerate(refined)]) ]
    return LinearSpline(all_knots, new_heights, new_slopes)
    
  @staticmethod
  def affine_combination_relu(weights, splines, bias):
    """Return the affine combination as above, but also take a relu"""
    combo = LinearSpline.affine_combination(weights, splines, bias)
    return combo.relu()
  
  @staticmethod
  def identity():
    return LinearSpline([0.0],[0.0],[1.0,1.0])
  
  @staticmethod
  def relu_identity(a,b):
    """Return the stard relu(x) function"""
    if a == 0:
      knots = [0.0]
      heights = [b]
      inf_slopes = [0.0,0.0]
    else:
      knots = [-b/a]
      heights = [0.0]
      inf_slopes = ([0.0,a] if a > 0 else [a,0.0])
    #print "Relu(",a,b,"): ", repr(LinearSpline(knots, heights, inf_slopes))
    return LinearSpline(knots, heights, inf_slopes)
  
  @staticmethod
  def from_nn(model, just_weights=False):
    """Turn a keras model into a linear spline.  The input and output must be
    a single node, and every layer must be a fully connected dense layer with
    relu activation.  Note this means each layer must have activation built in.
    Note this is SLOW and DEPRECATED in favor of from_nn_np"""
    w = (model if just_weights else model.get_weights())
    ls_layers = [ [ LinearSpline.relu_identity(w[0][0,i], w[1][i]) for i in range(len(w[1])) ] ]
    for i in range(2,len(w)-2,2):
      ls_layers.append( [ LinearSpline.affine_combination_relu( w[i][:,j].reshape( (-1,) ),
                                                                ls_layers[-1],
                                                                w[i+1][j] )
                                                                for j in range(w[i].shape[-1]) ] )
    ls_layers.append( [ LinearSpline.affine_combination( w[-2][:,0].reshape( (-1,) ),
                                                         ls_layers[-1],
                                                         w[-1][0] ) ] )
    return ls_layers
  
  @staticmethod
  def np_affine_combination_relu(weights, bias, knots, heights_in, inf_slopes_in):
    """Using numpy efficiently, compute an affine combination of the input splines and
    take a relu.  Note that all inputs are numpy arrays.  Here weights is a row vector or
    matrix in which each row gives the coefficients,
    and heights_in is a matrix where each row is a list of heights.  bias is a column
    vector giving the bias for each combination.  Knots is a row
    vector which gives the knots for all splines.  inf_slopes_in is a Nx2 array giving
    the inf slopes of all splines.
    See from_nn_np for how to use it"""
    #print "***"
    #print "Weights: ", weights
    #print "Bias: ", bias
    #print "Knots: ", knots
    #print "In heights: ", heights_in
    #print "In slopes: ", inf_slopes_in
    heights = np.dot(weights, heights_in) + bias
    inf_slopes = np.dot(weights, inf_slopes_in)
    #print "Heights: ", heights
    #print "Slopes: ", inf_slopes
    left_roots = ((-heights[:,0]/inf_slopes[:,0]) + knots[0]).reshape( (-1,) )
    left_roots = left_roots[ np.nonzero( (-np.inf < left_roots)*(left_roots < knots[0]) )[0] ]
    left_roots = np.sort(left_roots)
    #print "Left roots: ", left_roots
    #if len(left_roots) > 0:
    left_heights = (left_roots-knots[0])*inf_slopes[:,:1] + heights[:,:1]
    #print "Left heights shape: ", left_heights.shape
    new_knots = np.concatenate( (left_roots, knots[:1]) )
    new_heights = np.concatenate( (left_heights, heights[:,:1]), axis=1 )
    #else:
    #  new_knots = knots[:1]
    #  new_heights = heights[:,:1]
    
    for i in range(len(knots)-1):
      #print "New knots: ", new_knots
      #print "New heights: ", new_heights
      slopes = (heights[:,i+1:i+2] - heights[:,i:i+1]) / (knots[i+1] - knots[i])
      #print "Slopes: ", slopes, "shape: ", slopes.shape
      putative_roots = (knots[i] - (heights[:,i:i+1] / slopes)).reshape( (-1,) )
      #print "Putative roots: ", putative_roots, "shape: ", putative_roots.shape
      middle_knots = putative_roots[ np.nonzero( (knots[i] < putative_roots)*(putative_roots < knots[i+1]) )[0] ]
      middle_knots = np.sort(middle_knots)
      #print "Middle knots: ", middle_knots, " shape: ", middle_knots.shape
      #if len(middle_knots) > 0 :
      middle_heights = (middle_knots-knots[i])*slopes + heights[:,i:i+1]
      new_knots = np.concatenate( (new_knots, middle_knots, knots[i+1:i+2]) )
      new_heights = np.concatenate( (new_heights, middle_heights, heights[:,i+1:i+2]), axis=1 )
      #else:
      #  new_knots = np.concatenate( (new_knots, knots[i:i+1]) )
      #  new_heights = np.concatenate( (new_heights, heights[:,i:i+1]) )
    
    right_roots = ((-heights[:,-1]/inf_slopes[:,-1]) + knots[-1]).reshape( (-1,) )
    right_roots = right_roots[ np.nonzero( (knots[-1] < right_roots)*(right_roots < np.inf) )[0] ]
    right_roots = np.sort(right_roots)
    #print "Right roots: ", right_roots
    #if len(right_roots) > 0:
    right_heights = (right_roots-knots[-1])*inf_slopes[:,-1:] + heights[:,-1:]
    #print "Right heights shape: ", right_heights.shape
    new_knots = np.concatenate( (new_knots, right_roots) )
    new_heights = np.concatenate( (new_heights, right_heights), axis=1 )
    
    #print "Before relu:"
    #print "Knots: ", new_knots
    #print "Heights: ", new_heights
    #print "Slopes: ", inf_slopes
    
    new_heights = np.maximum( new_heights, 0.0 )
    inf_slopes[:,0] = np.minimum( inf_slopes[:,0], 0.0 )
    inf_slopes[:,1] = np.maximum( inf_slopes[:,1], 0.0 )
    
    #print "After relu:"
    #print "Heights: ", new_heights
    #print "Slopes: ", inf_slopes
    
    return (new_knots, new_heights, inf_slopes)
  
  
  
  @staticmethod
  def from_nn_np(model, just_weights=False):
    """Using numpy efficiently, produce a linear spline which is the output function
    of a neural network.  The network must have one input and one output"""
    w = (model if just_weights else model.get_weights())
    knots = np.array([0.0])
    heights = np.array([[0.0]])
    slopes = np.array([[1.0, 1.0]])
    for i in range(0, len(w)-2, 2):
      knots, heights, slopes = LinearSpline.np_affine_combination_relu(np.transpose(w[i]),
                                                                       w[i+1].reshape( (-1,1) ),
                                                                       knots, heights, slopes)
    tw = np.transpose(w[-2])
    final_slopes = np.dot(tw, slopes)
    final_heights = np.dot(tw, heights) + w[-1].reshape( (-1,1) )
    return LinearSpline(knots, final_heights[0], final_slopes[0])
  
  
  
  def __init__(self, knots, heights, inf_slopes):
    self.knots = [x for x in knots]
    self.heights = [x for x in heights]
    self.inf_slopes = [x for x in inf_slopes]
  
  def __repr__(self):
    return "LinearSpline(" + ','.join([repr(x) for x in [self.knots, self.heights, self.inf_slopes]]) + ')'
  
  def __str__(self):
    return "Linear spline with" + str(len(self.knots)) + "knots"
  
  def __sub__(self, other):
    return LinearSpline.affine_combination([1,-1], [self, other], 0)
  
  def __add__(self, other):
    return LinearSpline.affine_combination([1,1], [self, other], 0)
  
  def scalar_mul(self, scalar):
      return LinearSpline(self.knots, [scalar*h for h in self.heights], [scalar*s for s in self.inf_slopes])

  def __neg__(self):
    return LinearSpline( self.knots, [-x for x in self.heights], [-x for x in self.inf_slopes] )

  def plot(self, plot_roots=True):
    import matplotlib.pyplot as plt
    if len(self.knots) < 2:
      m = self.knots[0] - 0.5
      M = self.knots[0] + 0.5
    else:
      r = self.knots[-1] - self.knots[0]
      m = self.knots[0] - 0.1*r
      M = self.knots[-1] + 0.1*r
    if plot_roots:
      r = self.roots()
      m = min(r + [m])
      M = max(r + [M])
    pts = [m] + self.knots + [M]
    vals = [self.evaluate(m)] + self.heights + [self.evaluate(M)]
    plt.plot(pts, vals, color='blue')
    plt.plot(self.knots, self.heights, marker='o',
                                       linestyle='none',
                                       markerfacecolor='blue',
                                       markeredgecolor='none')
    if plot_roots:
      plt.plot([m,M],[0.0,0.0], color='#FF0000')
      plt.plot(r, [0]*len(r), marker='o', 
                              linestyle='none',
                              markerfacecolor='#FF0000',
                              markeredgecolor='none')
    plt.show()
  
  def refined(self, knots):
    """Return a new spline with the knots given; note that if knots is not a
    superset of self.knots, the spline won't be the same"""
    knot_heights = [self.evaluate(x) for x in knots]
    return LinearSpline(knots, knot_heights, self.inf_slopes)
  
  def slope(self, x):
    if x <= self.knots[0]:
      return self.inf_slopes[0]
    if x >= self.knots[-1]:
      return self.inf_slopes[1]
    for i in range(len(self.knots)):
      if self.knots[i] <= x and x < self.knots[i+1]:
        return (self.heights[i+1] - self.heights[i]) / (self.knots[i+1] - self.knots[i])
    print("ERROR: couldn't find slope")

  def restricted(self, x1, x2):
    """Return a linear spline restricted to the interval [x1, x2]; i.e. as
    close as possible to the original, but all knots are in that interval; note
    that if no knots lie in the range, it will make a knot in the middle, and 
    it will make knots on the endpoints"""
    new_inf_slopes = [x for x in self.inf_slopes]
    new_knots = []
    new_heights = []
    valid_knot_inds = [i for i,x in enumerate(self.knots) if x1 <= x and x <= x2]
    if len(valid_knot_inds) == 0:
      if self.knots[0] > x2:
        new_inf_slopes = [self.inf_slopes[0], self.inf_slopes[0]]
      else:
        new_inf_slopes = [self.inf_slopes[1], self.inf_slopes[1]]
      middle = 0.5*(x2-x1)
      return LinearSpline([middle], self.evaluate(middle), new_inf_slopes)
    new_inf_slopes = [ self.slope(x1), self.slope(x2) ]
    new_knots = [x1] + [self.knots[i] for i in valid_knot_inds] + [x2]
    new_heights = [self.evaluate(x1)] + [self.heights[i] for i in valid_knot_inds] + [self.evaluate(x2)]
    return LinearSpline(new_knots, new_heights, new_inf_slopes)

  def integrate(self, xmin, xmax):
    """Integrate self over a range"""
    # The integral of a linear function between (k1,h1) and (k2,h2) is h1*(k2-k1) + 0.5*(k2-k1)*(h2-h1)
    k_inds = [i for i in range(len(self.knots)) if xmin <= self.knots[i] and self.knots[i] <= xmax]
    eval_knots = [xmin] + [self.knots[i] for i in k_inds] + [xmax]
    eval_heights = [self.evaluate(xmin)] + [self.heights[i] for i in k_inds] + [self.evaluate(xmax)]
    s = 0.0
    for i in range(len(eval_knots)-1):
      k1, h1, k2, h2 = eval_knots[i], eval_heights[i], eval_knots[i+1], eval_heights[i+1]
      s += h1*(k2-k1) + 0.5*(k2-k1)*(h2-h1)
    return s

  def integrate_square(self, xmin, xmax):
    """Integrate the square of the spline."""
    # The integral of the spline between (k1,h1) and (k2,h2) is (1/3)*(h1^2 + h1*h2 + h2^2)*(k2 - k1)
    k_inds = [i for i in range(len(self.knots)) if xmin <= self.knots[i] and self.knots[i] <= xmax]
    eval_knots = [xmin] + [self.knots[i] for i in k_inds] + [xmax]
    eval_heights = [self.evaluate(xmin)] + [self.heights[i] for i in k_inds] + [self.evaluate(xmax)]
    s = 0.0
    for i in range(len(eval_knots)-1):
      k1, h1, k2, h2 = eval_knots[i], eval_heights[i], eval_knots[i+1], eval_heights[i+1]
      s += (h1*h1 + h1*h2 + h2*h2)*(k2 - k1)
    s *= (1.0/3.0)
    return s

  def roots(self):
    """Return all roots of the linear spline.  Note there are degenerate cases
    in which the spline has an interval at height 0; this function
    tries to alert you"""
    ans = []
    if (self.inf_slopes[0] > 0 and self.heights[0] > 0) or \
       (self.inf_slopes[0] < 0 and self.heights[0] < 0):
      ans.append( (-self.heights[0]/self.inf_slopes[0]) + self.knots[0] )
      #print "Root from inf_slope[0]:", ans[-1]
    if self.inf_slopes[0] == 0 and self.heights[0] == 0:
      print("Interval of roots")
    for i in range(len(self.knots)-1):
      if self.heights[i] == 0:
        ans.append(self.knots[i])
        #print "Root at knot:",i,": ", ans[-1]
        if self.heights[i+1] == self.heights[i]:
          print("Interval of roots")
      elif self.heights[i]*self.heights[i+1] < 0:
        r = abs(self.heights[i+1] / self.heights[i])
        d = self.knots[i+1] - self.knots[i]
        ans.append( self.knots[i] + (d/(1+r)) )
        #print "Root between knots:",i,i+1,": ", ans[-1]
    if self.heights[-1] == 0:
      if self.inf_slopes[1] == 0:
        print("Interval of roots")
      ans.append(self.knots[-1])
      #print "Root at last knot: ", ans[-1]
    if (self.inf_slopes[1] < 0 and self.heights[-1] > 0) or \
       (self.inf_slopes[1] > 0 and self.heights[-1] < 0):
      ans.append( (-self.heights[-1]/self.inf_slopes[1]) + self.knots[-1] )
      #print "Root a from inf_slope[1]: ", ans[-1]
    return ans
  
  def evaluate(self, x):
    """Evaluate the spline at the given point x"""
    if x < self.knots[0]:
      return (x-self.knots[0])*self.inf_slopes[0] + self.heights[0]
    for i in range(len(self.knots)-1):
      if x == self.knots[i]:
        return self.heights[i]
      if x < self.knots[i+1]:
        if self.knots[i] != self.knots[i+1]:
          t = (x - self.knots[i])/(self.knots[i+1]-self.knots[i])
          return (1-t)*self.heights[i] + t*self.heights[i+1]
    if x == self.knots[-1]:
      return self.heights[-1]
    return (x-self.knots[-1])*self.inf_slopes[1] + self.heights[-1]
  
  def relu(self):
    """Return a new spline which is the relu of self"""
    roots = self.roots()
    rootset = set(roots)
    all_points = sorted(list(set(roots + self.knots)))
    refined = self.refined(all_points)
    new_slopes = [ (0.0 if refined.inf_slopes[0] > 0 else refined.inf_slopes[0]),
                   (0.0 if refined.inf_slopes[1] < 0 else refined.inf_slopes[1]) ]
    new_knot_inds = [i for i,x in enumerate(refined.knots) if (x in rootset or refined.heights[i] >= 0.0)]
    new_knots = [refined.knots[i] for i in new_knot_inds]
    #DANGER ROUNDING because it makes sense to force roots to have height 0?
    new_heights = [(0.0 if refined.knots[i] in rootset else refined.heights[i]) for i in new_knot_inds]
    if len(new_knots) == 0:
      return LinearSpline([0.0],[0.0],[0.0,0.0])
    return LinearSpline(new_knots, new_heights, new_slopes)



def random_affine_combination(n, kind='reluniform'):
  if kind == 'reluniform':
    S = [LinearSpline.relu_identity(np.random.uniform(-1,1), np.random.uniform(-1,1)) for i in range(n)]
  elif kind == 'knotuniform':
    S = [LinearSpline([np.random.uniform(-1,1)], [np.random.uniform(-1,1)], np.random.uniform(-1,1,(2,))) for i in range(n)]
  w = [np.random.uniform(-1,1) for x in S]
  v = [x.evaluate(-1.2) for x in S]
  #print "Weights, values at -1.2:", zip(w,v)
  print("True affine combination at -1.2:", sum([x*y for (x,y) in zip(w,v)]))
  s = LinearSpline.affine_combination(w, S, 0)
  print("Affine combination value:", s.evaluate(-1.2))
  return s

def random_net(layer_widths):
  weights = [ np.random.uniform(-1,1,(1,layer_widths[0])),
              np.random.uniform(-1,1,(layer_widths[0],)) ]
  for i in range(1, len(layer_widths)):
    weights.extend( [ np.random.uniform(-1,1,(layer_widths[i-1], layer_widths[i])),
                      np.random.uniform(-1,1,(layer_widths[i],)) ] )
  weights.extend( [ np.random.uniform(-1,1,(layer_widths[-1], 1)),
                    np.random.uniform(-1,1,(1,)) ] )
  return LinearSpline.from_nn_np(weights, just_weights=True)

def f(x):
  np.random.seed(sum(map(ord, os.urandom(10))))
  return (x[0], x[1], len(random_net(x[2])[-1][-1].knots))

def random_net_splines(W, D, trials):
  data = [(x,y,x*[y]) for x in D for y in W for t in range(trials)]
  #f = lambda x:(x[0], x[1], len(random_net(x[2])[-1][-1].knots))
  from multiprocessing import Pool
  p = Pool()
  ans = p.map(f, data)
  return ans














