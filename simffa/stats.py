import numpy as np

from scipy.stats import ss, betai, distributions, rankdata

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis


def prod(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0]*prod(x[1:])
        
def pearsonr(x, y):
    """
    generalized from scipy.stats.pearsonr 
    """
    # x and y should have same length.
    x_shape = x.shape
    if len(x_shape) > 1:
        x = x.reshape((x_shape[0],prod(x_shape[1:])))
    
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean(0)
    my = y.mean()
    xm, ym = x-mx, y-my
    
    r_num = n*np.dot(xm.T,ym)
    r_den = n*np.sqrt(ss(xm)*ss(ym))
    
    r = (r_num / r_den)

    # Presumably, if r > 1, then it is only some small artifact of floating
    # point arithmetic.
    r = np.minimum(r, 1.0)
    df = n-2

    # Use a small floating point value to prevent divide-by-zero nonsense
    # fixme: TINY is probably not the right value and this is probably not
    # the way to be robust. The scheme used in spearmanr is probably better.
    TINY = 1.0e-20
    t = r*np.sqrt(df/((1.0-r+TINY)*(1.0+r+TINY)))
    prob = betai(0.5*df,0.5,df/(df+t*t))
    
    if len(x_shape) > 1:
        r = r.reshape(x_shape[1:])
        prob = prob.reshape(x_shape[1:])
    
    return r,prob
    

def spearmanr(a, b=None, axis=0):
    """
    generalized from scipy.stats.spearmanr
    """
    a_shape = a.shape
    if len(a_shape) > 1:
        a = a.reshape((a_shape[0],prod(a_shape[1:])))

    a, axisout = _chk_asarray(a, axis)
    ar = np.apply_along_axis(rankdata,axisout,a)
    
    br = None
    if not b is None:
        b, axisout = _chk_asarray(b, axis)
        br = np.apply_along_axis(rankdata,axisout,b)
    n = a.shape[axisout]
    if len(ar.shape) == 2:
        rs0 = np.corrcoef(ar.T,br)[:,-1][:-1]
        rs = np.ones((2,2,ar.shape[1]))
        rs[0,1,:] = rs0
        rs[1,0,:] = -rs0
    else:
        rs = np.corrcoef(ar,br,rowvar=axisout)

    t = rs * np.sqrt((n-2) / ((rs+1.0)*(1.0-rs)))
    prob = distributions.t.sf(np.abs(t),n-2)*2
    
    if rs.shape[:2] == (2,2):
        rs, prob = rs[0,1], prob[1,0]

    if len(a_shape) > 1:
        rs = rs.reshape(a_shape[1:])
        prob = prob.reshape(a_shape[1:])
    
    return rs, prob
        
