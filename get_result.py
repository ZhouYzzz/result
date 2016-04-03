import datetime
import numpy as np

def test_on_dataset(
	re_identifier,
	re_identifier_args,
	dataset_name='VIPeR',
	display=False,
	display_rank=200,):
	'''Test a method on a specified dataset'''
	log('Test started')
	# read dataset
	a, b, _ = read_dataset(dataset_name)
	result = re_identifier(a, b, **re_identifier_args)
	rank = get_rank(result)
	import matplotlib.pyplot as plt
	plt.plot(rank[:display_rank], 'o-')
	plt.show()

def get_rank(result):
	rank = np.zeros(result.shape[1])

	for i in xrange(result.shape[1]):
		ranki = np.where(result[i, :].argsort()==i)[0][0]
		rank[ranki:] += 1

	rank = rank/result.shape[1]
	return rank

def read_dataset(dataset_name='VIPeR'):
	'''return a list of test image pairs' path'''
	import pandas as pd
	import cv2
	if dataset_name=='VIPeR':
		log('Read dataset [ %s ]'%dataset_name)
		VIPeR_path = '/home/gpu/zhouyz/VIPeR'
		p_a = pd.read_csv(VIPeR_path+'/cam_a.txt',sep=' ',header=None)
		p_b = pd.read_csv(VIPeR_path+'/cam_b.txt',sep=' ',header=None)
		assert p_a.shape == p_b.shape
		log('All [ %d ] pairs of test images'%p_a.shape[0])
		p_a = p_a.iloc[:,0].astype(str).values
		p_b = p_b.iloc[:,0].astype(str).values
		p_a = VIPeR_path + p_a
		p_b = VIPeR_path + p_b
	elif dataset_name=='prid':
		log('Read dataset [ %s ]'%dataset_name)
		prid_path = '/home/gpu/zhouyz/prid_2011/single_shot'
		p_a = pd.read_csv(prid_path+'/cam_a.txt',sep=' ',header=None)
		p_b = pd.read_csv(prid_path+'/cam_b.txt',sep=' ',header=None)
		assert p_a.shape == p_b.shape
		log('All [ %d ] pairs of test images'%p_a.shape[0])
		p_a = p_a.iloc[:,0].astype(str).values
		p_b = p_b.iloc[:,0].astype(str).values
		p_a = prid_path + p_a
		p_b = prid_path + p_b
	else:
		log('Error dataset name [ %s ]'%dataset_name)
		exit()

	assert cv2.imread(p_a[0]) is not None
	assert cv2.imread(p_b[0]) is not None
	num_pairs = p_a.shape[0]
	return p_a, p_b, num_pairs

def re_identifier_deep(list_a, list_b, caffe_model, caffe_weights, model_name):
	'''pass list A and B to deep net and related mothods'''
	# set_up_caffe()
	model = load_model(model_name)
	net = load_net(caffe_model, caffe_weights)
	# print model.support_vectors_
	# exit()

	num_a = list_a.shape[0]
	num_b = list_b.shape[0]

	import cv2
	import numpy as np
	# from time import time as t
	for im in list_a:
		# print im
		res = forward(net, im)
		try:
			res_a = np.vstack((res_a, res))
		except:
			res_a = res

	for im in list_b:
		# print im
		res = forward(net, im)
		try:
			res_b = np.vstack((res_b, res))
		except:
			res_b = res

	print res_a[0,:].shape
	print res_b[0,:].shape

	result = np.zeros([num_a, num_b])
	for ia in xrange(num_a):
		# dis = res_b - res_a[ia,:]
		# print dis
		# print dis
		# print dis
		# prod = model.predict_proba(dis)
		# print prod, prod.shape
		dis = np.linalg.norm(res_b - res_a[ia,:],ord=2,axis=1)
		# print dis.tolist()
		result[ia,:] = dis.T

	# result = np.zeros([num_a, num_b])

	return result

def forward(net, im, blobs=['feature7']):
	import cv2
	im = cv2.imread(im)
	assert im is not None
	im = cv2.resize(im, (128,256))
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)

	try:
		res = net.forward_all(blobs=blobs, data=im)
	except:
		log('Forward error')
		exit()

	for layer in blobs:
		try:
			res_all = np.hstack((res_all, res[layer].reshape(-1)))
		except:
			res_all = res[layer].reshape(-1)

	return res_all

def load_model(model_name):
	import cPickle as pickle
	log('Load Model')
	try:
		return pickle.load(open(model_name))
	except:
		log('Load Model **Failed**')
		exit()

def set_up_caffe():
	import sys
	sys.path.insert(0, '/home/gpu/zhouyz/caffe/python')
	import warnings
	warnings.filterwarnings('ignore')
	import caffe
	caffe.set_mode_gpu()
	caffe.set_device(0)
	log('Caffe set up successfully')
	return caffe

def load_net(caffe_model, caffe_weights):
	log('Load Net')
	try:
		return caffe.Net(caffe_model, caffe_weights, caffe.TEST)
	except:
		log('Load Net **Failed**')
		exit()

def log(str):
	print datetime.datetime.now(), ':', str


re_identifier_args = {
		'caffe_model': '/home/gpu/zhouyz/SqueezeNet/SqueezeNet_v1.0/deploy_cuhk.prototxt',
		'caffe_weights': '/home/gpu/zhouyz/SqueezeNet/SqueezeNet_v1.0/snapshots/train_iter_1502.caffemodel',
		'model_name': 'svm'
}

if __name__ == '__main__':

	caffe = set_up_caffe()

	# read_dataset()
	test_on_dataset(
		re_identifier_deep, 
		re_identifier_args,
		dataset_name='prid'
		)