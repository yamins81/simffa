import numpy as np
import scipy.stats as sp_stats
import os
import tables as tbl

from pyll import scope
import pyll.stochastic as stochastic
import pyll.base 
import skdata.larray as larray

import simffa.simffa_params as sp
import simffa.simffa_bandit as sb

import simffa.simffa_fboDat as fbo
import simffa.simffa_facegens as fgs 
import simffa.simffa_mtDat as mtdat

from mvpa2.suite import SimpleSOMMapper
import cPickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model_param = sp.l3_params
conf = stochastic.sample(model_param, np.random.RandomState(0))

record = fgs.fgs_bandit_evaluate3(conf)
maps = record['results']['maps']
map_types = ['id', 'pose', 'expression']
for m in map_types:
	plt.figure()
	plt.imshow(maps[m])
	plt.savefig(m + '_map.png')

with open (os.path.join('maps.pkl'), 'w') as _f:
		cPickle.dump(maps, _f)  


model_param = sp.l3_params
conf = stochastic.sample(model_param, np.random.RandomState(0))

dataset = fbo.FaceBodyObject20110803()
imgs, labels = dataset.get_images()
features = sb.get_features(imgs, conf)
fsi = fbo.getFSI(features, labels)



# load MTurk image data - with nInvar number of invariant reps
def initMT(nInvar=0):
	dataset = mtdat.MTData_Feb222013()
	imgs,labels = dataset.get_images(nInvar)
	N = labels.shape
	print 'Loading ' + str(N[0]) + ' images.'
	return imgs, labels, dataset

# load pre-saved MTurk image data
def loadMT(fn):
	h5file = tbl.openFile(fn, 'r')
	imgs = h5file.root.imgs
	labels = h5file.root.labels
	dataset = h5file.root.dataset
	imgs = h5file.root.imgs
	h5file.close()
	return imgs, labels, dataset

## per model functions ##

# save results per model
def saveResult(h5file, result):
	h5file.createGroup(h5file.root, 'result', title='result')
	h5file.createArray(h5file.root.result, 'psyCorr_mu', result['psyCorr_mu'])
	h5file.createArray(h5file.root.result, 'psyCorr_hist_n', result['psyCorr_hist_n'])
	h5file.createArray(h5file.root.result, 'psyCorr_hist_x', result['psyCorr_hist_x'])
	h5file.createArray(h5file.root.result, 'psyCorr_cluster', result['psyCorr_cluster'])
	h5file.createArray(h5file.root.result, 'psyCorr_blob', result['psyCorr_blob'])

# plots per model
def makeplots(result, fname):
	prop = [0]*3
	prop[0] = np.float(result['psyCorr_cluster'])
	prop[1] = np.float(result['psyCorr_blob'])
	prop[2] = np.abs(result['psyCorr_mu']).ravel().mean(0)
	
	if fname is not None:
		plt.figure()

		plt.subplot(1,2,1)
		plt.imshow(result['psyCorr_mu']), plt.colorbar()
		plt.xlabel('cluster = ' + str(prop[0]))
		plt.ylabel('inv peakness = ' + str(prop[1]))
		
		plt.subplot(1,2,2)
		plt.plot(result['psyCorr2_hist_x'] ,result['psyCorr2_hist_n'] )
		plt.title('mean rho = ' + str(prop[2]))
		plt.savefig(fname + '.png')  
	return prop


## population functions ##

# sample models (if overwrite) and compute psyCorr + properties
def sampleModels(model_param, N, imgs, labels, dataset, overwrite=False):
	print 'Sampling ' + str(N) + ' ' + model_param['tag']  + ' models:'
	model_prop = []
	Nim = labels.shape[0]

	for i in range(N):
		outname = 'model' + str(Nim) + model_param['tag'] + '_' + str(i)
		fileExist = os.path.isfile(basedir+outname + '.h5')
		conf = stochastic.sample(model_param, np.random.RandomState(i))

		if (not fileExist) or overwrite:
			print basedir + outname + ' doesnt exist'
			features = sb.get_features(imgs, conf, False, basedir, outname)
		
		h5file = tbl.openFile(basedir + outname + '.h5', mode = 'a')
		features = h5file.root.features
		result = sb.evaluate_psyFace(None, features, labels, False)
		# fsiResult = evaluate_FSI(conf, features=None, labels=None, train=False)

		saveResult(h5file, result)
		h5file.close()
		curr_prop = makeplots(result, None)
		model_prop.append(curr_prop)		

		# regress_result, rsq = sb.regression_traintest(dataset,features,labels,0,200,100,5)
		# model_prop[i][2] = rsq

	return model_prop

def computeProps(model_param, pop_outname, N, Nim):
	import simffa_analysisFns as sa
	print 'Recomputing ' + str(N) + ' ' + model_param['tag']  + ' models properties:'
	model_prop = []
	
	for i in range(N):
		prop = [0]*3
		outname = 'model' + str(Nim) + model_param['tag'] + '_' + str(i)
		h5file = tbl.openFile(basedir + outname + '.h5', mode = 'a')
		features = h5file.root.features
		psyCorr_mu = h5file.root.result.psyCorr_mu

		prop[0] = sa.getClusterSize(psyCorr_mu)
		prop[1] = sa.getBlobiness(psyCorr_mu)
	
		prop[2] = np.abs(psyCorr_mu[~np.isnan(psyCorr_mu)]).ravel().mean(0)
		h5file.close()
		model_prop.append(prop)

	h5file = tbl.openFile(pop_outname, 'a')
	# h5file.removeNode(h5file.root, 'prop')
	h5file.createArray(h5file.root, 'prop', model_prop)
	h5file.close()


def plotPopulationResults(model_prop, model_param):
	fname_tag = 'fig/' + model_param['tag']
	model_prop = np.array(model_prop)
	
	x = model_prop[:,0]
	y = model_prop[:,1]
	z = model_prop[:,2]
	
	t = ~np.isnan(x) & ~np.isnan(z) & ~np.isnan(z)
	plt.figure()
	plt.plot(x[t], z[t], 'o')
	plt.xlabel('Relative cluster size')
	plt.ylabel('Mean rho')
	plt.title(str(np.corrcoef(x[t],z[t])[0][1]))
	plt.savefig(fname_tag + '_cluster.png')

	t = ~np.isnan(y) & ~np.isnan(z) & ~np.isnan(z)
	plt.figure()
	plt.plot(y[t], z[t], 'o')
	plt.xlabel('Inverse peakness')
	plt.ylabel('Mean rho')
	plt.title(str(np.corrcoef(y[t],z[t])[0][1]))	
	plt.savefig(fname_tag + '_peak.png')


def plotSortedModels(params, pop_outname, N, Nim):

	h5file = tbl.openFile(pop_outname)
	prop = h5file.root.prop
	values = prop[:,1]
	s_i = [i[0] for i in sorted(enumerate(values), key=lambda x:x[1])]

	count = 0
	stepSize = np.ceil(len(s_i)/10)
	for i in range(0,len(s_i), stepSize):
		s_oi = s_i[i]
		fn = basedir + 'model' + str(Nim) + params['tag'] + '_' + str(s_oi) + '.h5'
		m = tbl.openFile(fn)
		psyCorr = m.root.result.psyCorr_mu
		count = count+1
		plt.subplot(2,5,count)
		plt.imshow(psyCorr)
		plt.title('{:.4}'.format(values[s_oi]))
		plt.colorbar()
		m.close()

	fign = params['tag'] + 'model_' + str(Nim) + '_sorted.png'
	plt.savefig(fign)
	plt.close()
	h5file.close()


def main():
	nInvar = 0;
	imgs, labels, dataset = initMT(nInvar)
	Nim = labels.shape[0]

	params = sp.l1_params
	# conf = stochastic.sample(model_param, np.random.RandomState(i))
	Nmodels = 20
	overwrite = False
	global basedir 

	pop_outname = params['tag'] + 'model_' + str(Nim) + '_pop.h5'
	basedir = params['tag']  + 'model_' + str(Nim) + '/'
	os.mkdir(basedir)

	model_prop = sampleModels(params, Nmodels, imgs, labels, dataset, overwrite)
	plotPopulationResults(model_prop, params)

	h5file = tbl.openFile(pop_outname, 'a')
	h5file.createArray(h5file.root, 'imgs', imgs)
	h5file.createArray(h5file.root, 'labels', labels)
	h5file.createArray(h5file.root, 'prop', model_prop)
	h5file.close()


# main()


