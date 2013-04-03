
import numpy as np
import scipy.stats as sp_stats
import scipy.ndimage.measurements as sp_meas
from scipy.stats.stats import pearsonr
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from operator import mul
from sklearn.neighbors import NearestNeighbors

def intersect_index(a,b):
	t = [i for i,item in enumerate(a) if item in b]
	return t

def topogProduct_getQ(f_dist, f_ind, a_dist, a_ind, k):
	fdi_fnn = reduce(mul, f_dist[:,:k].T, 1)
	adi_ann = reduce(mul, a_dist[:,:k].T, 1)
	n = a_ind.shape[0]
	Q = [0] * n
	for i in range(n):
	    ann = a_ind[i,:k]
	    fnn = f_ind[i,:k]
	    t = intersect_index(f_ind[i,:], ann)
	    fdi_ann = reduce(mul, f_dist[i,t], 1)
	    t = intersect_index(a_ind[i,:], fnn)
	    adi_fnn = reduce(mul, a_dist[i,t], 1)
	    
	    Q1 = (fdi_ann+sys.float_info.epsilon)/(fdi_fnn[i] +sys.float_info.epsilon)
	    Q2 = (adi_fnn+sys.float_info.epsilon)/(adi_ann[i] + 1+sys.float_info.epsilon)
	    Q[i] = np.log(Q1*Q2) / (2*k)
	return np.array(Q).mean(0)

# measure of topological preservation (neighbours in anatomical space remain neighbours in feature space)
def topographicProcuct(X):
	anat_neigh = NearestNeighbors()
	feat_neigh = NearestNeighbors()
	n = X.shape[0]
	zeros = np.array([[0 for i in range(n)] for j in range(n)])
	ix = np.array([[i for i in range(n)] for j in range(n)])
	jx = np.array([[j for i in range(n)] for j in range(n)])

	x = np.array([X.ravel(), zeros.ravel()]).T
	ax = np.array([ix.ravel(),jx.ravel()]).T

	nPoints = n #should sample more random points
	randPoints = [np.random.randint(0,n**2) for r in xrange(nPoints)]
	x = x[randPoints,:]
	ax = ax[randPoints,:]

	print 'computing nearest neighbours'
	feat_neigh.fit(x)
	f_dist,f_ind = feat_neigh.kneighbors(x, n_neighbors=nPoints, return_distance=True)
	f_dist = f_dist[:,1:]
	f_ind = f_ind[:,1:]

	anat_neigh.fit(ax)
	a_dist, a_ind = anat_neigh.kneighbors(ax, n_neighbors=nPoints, return_distance=True)
	a_dist = a_dist[:,1:]
	a_ind = a_ind[:,1:]
	print 'computing topog product'
	P = [0] * (nPoints-1)
	for k in range(np.int(n/2)):
		if k == 0:
			continue
		P[k] = P[k-1] + topogProduct_getQ(f_dist, f_ind, a_dist, a_ind, k)
	return np.array(P).mean(0)


def getPearsonCorr2D(X,Y):
	# X = np.array(X)
	# Y = np.array(Y)
	fs = X.shape
	num_feats = fs[1]*fs[2]

	X2 = X.reshape(fs[0], num_feats)
	Y2 = Y.reshape(fs[0], 1)
	XY = np.hstack((X2,Y2))
	R = np.corrcoef(XY.T)
	R = R[:num_feats,num_feats]
	R1 = R.reshape(fs[1], fs[2])
	return R1

# more efficient way to compute correlation between each "column" of 4 dimensional matrix X and a vector Y
def getPearsonCorr(X,Y):
	X = np.array(X)
	Y = np.array(Y)
	fs = X.shape
	num_feats = fs[1]*fs[2]*fs[3]

	X2 = X.reshape(fs[0], num_feats)
	Y2 = Y.reshape(fs[0], 1)
	XY = np.hstack((X2,Y2))
	R = np.corrcoef(XY.T)
	R = R[:num_feats,num_feats]
	R1 = R.reshape(fs[1], fs[2], fs[3])
	return R1

# how correlated is the distance in anatomical space to the distance in feature space
def getTopographicOrg(X):
	X = np.array(X) 
	fs = X.shape
	num_pairs = fs[0]*fs[1] #probably should sample more points than this
	AF_dist = []
	for i in range(num_pairs):
		i1 = np.random.randint(fs[0])
		j1 = np.random.randint(fs[1])
		i2 = np.random.randint(fs[0])
		j2 = np.random.randint(fs[1])
		aDist = np.sqrt((i1-i2)**2 +  (j1-j2)**2)
		fDist = abs(X[i1,j1] - X[i2,j2]) / (abs(X[i1,j1] + X[i2,j2]) + sys.float_info.epsilon)
		if (fDist < 0.01) | (fDist > 0.99):
			continue
		AF_dist.append([aDist, fDist])
	AF_dist = np.array(AF_dist)
	r,p = pearsonr(AF_dist[:,1], AF_dist[:,0])

	return r

# if the heat map were a probability distribution, what would its standard deviation be
# (or just the average weighted distance from the center of mass)
def getClusterSize(X):
	X = np.array(X) + 1 # make sure all are positive
	fs = X.shape
	cm = sp_meas.center_of_mass(X) #mean of cluster
	dist_cm  = [[ np.sqrt((cm[0]-j)**2+(cm[1]-i)**2)  for i in range(fs[1])] for j in range(fs[0])]
	weightedDist = dist_cm * X
	clusterSize = weightedDist.sum() / np.array(X).sum()
	relativeCluterSize = clusterSize / fs[0]
	return relativeClusterSize

# how far from flat is the map
def getBlobiness(x):
	x = x[~np.isnan(x)]
	x0 = np.abs(x - x.mean()) + sys.float_info.epsilon
	# y = -np.log(x0/len(x.ravel())).mean()
	y = (x0/len(x.ravel())).mean()
	return y


def getAsymmetryIndex(X):
	cm = sp_meas.center_of_mass(X)
	peak_val = X.max()
	peak_i,peak_j = np.unravel_index(X.argmax(), X.shape)
	width = np.sqrt(sum(sum(X > X.mean())))
	ASI = np.sqrt((cm[0]-peak_i)**2 + (cm[1]-peak_j)**2) / width

	return ASI, width


def testAsymmetryIndex():	

	plt.figure()
	for ii in range(10):

		if ii < 5:
			mu = [np.random.normal(0) for i in range(2)] 
			cov  = [[np.random.normal((ii+1)*100) for i in range(2)] for j in range(2)]
			cov = np.array(cov)
			cov[1][0] = 0
			cov[0][1] = 0
			x = np.random.multivariate_normal(mu, cov, 10000)
			n, nx, ny = np.histogram2d(x[:,0],x[:,1],50, [[-50,50], [-50, 50]])
		else:
			n = np.array([[np.random.random() for i in range(50)] for j in range(50)])

		width = (topographicProcuct(n))
		# width = getBlobiness(n)
		# width = getTopographicOrg(n)

		plt.subplot(2,5,ii+1)
		plt.imshow(n)
		plt.title('{:.4}'.format(width))
		print 'test' + str(ii)
	print 'saving fig/tp_test.png'
	plt.savefig('fig/tp_test.png')

def testCorr():
	N = 100;
	N1 = 5;
	N2 = 7;
	N3 = 10;
	X = np.array([[[[np.random.random() for i in range(N1)] for j in range(N2)] for k in range(N3)] for l in range(N)])
	Y = np.array([np.random.random() for k in range(N)])
	R1 = np.array([[[ pearsonr(X[:,j,k,l],Y)[0] for l in range(N1)] for k in range(N2)] for j in range(N3)])
	R2 = getPearsonCorr(X,Y)
	print R1
	print ''
	print R2
	return R1, R2

# testAsymmetryIndex()