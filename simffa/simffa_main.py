import numpy as np
import scipy.stats as sp_stats
import os
import tables as tbl

from pyll import scope
import pyll.stochastic as stochastic
import pyll.base 
import skdata.larray as larray

import simffa_params as sp
import simffa_bandit as sb

import simffa_mtDat as mtdat
import simffa_fboDat as fbo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def initMT(nInvar=0):
	dataset = mtdat.MTData_Feb222013()
	imgs,labels = dataset.invariant_img_classification_task(nInvar)
	N = labels.shape
	print 'Loading ' + str(N[0]) + ' images.'
	return imgs, labels, dataset

def loadMT(fn):
	h5file = tbl.openFile(fn, 'r')
	imgs = h5file.root.imgs
	labels = h5file.root.labels
	dataset = h5file.root.dataset
	imgs = h5file.root.imgs
	h5file.close()
	return imgs, labels, dataset

def saveResult(h5file, result):
	h5file.createGroup(h5file.root, 'result', title='result')
	h5file.createArray(h5file.root.result, 'psyCorr_mu', result['psyCorr_mu'])
	h5file.createArray(h5file.root.result, 'psyCorr_hist_n', result['psyCorr_hist_n'])
	h5file.createArray(h5file.root.result, 'psyCorr_hist_x', result['psyCorr_hist_x'])
	h5file.createArray(h5file.root.result, 'psyCorr_cluster', result['psyCorr_cluster'])
	h5file.createArray(h5file.root.result, 'psyCorr_blob', result['psyCorr_blob'])

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

def sampleModels(model_param, N, imgs, labels, dataset, overwrite=False):
	print 'Sampling ' + str(N) + ' ' + model_param['tag']  + ' models:'
	model_prop = []
	Nim = labels.shape[0]
	# basedir = model_param['tag']  + 'model_' + str(Nim) + '/'

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

def plotPopulationResults(model_prop, model_param):
	fname_tag = 'fig/' + model_param['tag']
	model_prop = np.array(model_prop)
	
	x = model_prop[:,0]
	y = model_prop[:,1]
	z = model_prop[:,2]
	
	t = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
	x = x[t]
	y = y[t]
	z = z[t]

	plt.figure()
	plt.plot(x, z, 'o')
	plt.xlabel('Relative cluster size')
	plt.ylabel('Mean rho')
	plt.title(str(np.corrcoef(x,z)[0][1]))
	plt.savefig(fname_tag + '_cluster.png')

	plt.figure()
	plt.plot(y[y<40], z[y<40], 'o')
	plt.xlabel('Inverse peakness')
	plt.ylabel('Mean rho')
	plt.title(str(np.corrcoef(y[y<40],z[y<40])[0][1]))	
	plt.savefig(fname_tag + '_peak.png')

# def plotSummary(model_prop):
# 	[i[0] for i in sorted(enumerate(prop[:,1]), key=lambda x:x[1])]
	

def main():
	nInvar = 0;
	imgs, labels, dataset = initMT(nInvar)
	Nim = labels.shape[0]

	params = sp.l1_params
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


