
import numpy as np
import scipy.stats as sp_stats
import scipy.ndimage.measurements as sp_meas
from scipy.stats.stats import pearsonr
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def getPearsonCorr2D(X,Y):
	X = np.array(X)
	Y = np.array(Y)
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

def getClusterSize(X):
	X = np.array(X) #+ 1 # make sure all are positive
	fs = X.shape
	cm = sp_meas.center_of_mass(X) #mean of cluster
	dist_cm  = [[ np.sqrt((cm[0]-j)**2+(cm[1]-i)**2)  for i in range(fs[1])] for j in range(fs[0])]
	weightedDist = dist_cm * X
	clusterSize = weightedDist.sum() / np.array(X).sum()
	return clusterSize

def getBlobiness(x):
	x = x[~np.isnan(x)]
	x0 = np.abs(x - x.mean()) + sys.float_info.epsilon
	y = -np.log(x0/len(x.ravel())).mean()
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

		# width = np.int(getClusterSize(n))
		width = getBlobiness(n)

		plt.subplot(2,5,ii+1)
		plt.imshow(n)
		plt.title('{:.4}'.format(width))
	
	plt.savefig('fig/peak_test.png')

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

# R1,R2 = testCorr()

testAsymmetryIndex()