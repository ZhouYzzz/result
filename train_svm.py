from sklearn import svm
clf = svm.LinearSVC()
# probability=True
from get_result import *
caffe = set_up_caffe()

net = caffe.Net(re_identifier_args['caffe_model'], 
	re_identifier_args['caffe_weights'], caffe.TEST)

import pandas as pd
list = pd.read_csv('labeled.txt', sep=' ', header=None)
images = list.iloc[:,0].astype(np.str).values
labels = list.iloc[:,1].values

num = images.shape[0]
log('Total [ %d ] images'%num)

import numpy as np

CLASS = 1000
for i in xrange(CLASS):
	if i%100==0:log('Handel [ %d / %d] classes'%(i, CLASS))
	c1_ = np.random.choice(images[labels==i],3,False)
	c2_ = np.random.choice(images[labels!=i],1,False)
	same = c1_[:2]
	# print same
	diff = [c1_[2], c2_[0]]
	# print diff
	
	sf1 = forward(net, same[0], ['ip2'])
	sf2 = forward(net, same[1], ['ip2'])
	sd = np.absolute(sf1-sf2)
	# print sd
	# print np.linalg.norm(sd,ord=2)

	df1 = forward(net, diff[0], ['ip2'])
	df2 = forward(net, diff[1], ['ip2'])
	dd = np.absolute(df1-df2)
	# print dd
	# print np.linalg.norm(dd,ord=2)

	try:
		sd_all = np.vstack((sd_all, sd))
	except:
		sd_all = sd

	try:
		dd_all = np.vstack((dd_all, dd))
	except:
		dd_all = dd

dif = np.vstack((sd_all, dd_all))
sim = np.hstack((np.ones(sd_all.shape[0]),np.zeros(dd_all.shape[0])))

# print dif
# print sim

# print np.linalg.norm(dif,ord=2,axis=1)
log('SVM fit')
clf.fit(dif,sim)

print clf.score(dif,sim)


'''
TEST
'''





for i in xrange(1000,1100):
	if i%100==0:log('Handel [ %d / %d] classes'%(i, CLASS))
	c1_ = np.random.choice(images[labels==i],3,False)
	c2_ = np.random.choice(images[labels!=i],1,False)
	same = c1_[:2]
	# print same
	diff = [c1_[2], c2_[0]]
	# print diff
	
	sf1 = forward(net, same[0], ['ip2'])
	sf2 = forward(net, same[1], ['ip2'])
	sd = np.absolute(sf1-sf2)
	# print sd
	# print np.linalg.norm(sd,ord=2)

	df1 = forward(net, diff[0], ['ip2'])
	df2 = forward(net, diff[1], ['ip2'])
	dd = np.absolute(df1-df2)
	# print dd
	# print np.linalg.norm(dd,ord=2)

	try:
		sd_all = np.vstack((sd_all, sd))
	except:
		# print "initialize"
		sd_all = sd

	try:
		dd_all = np.vstack((dd_all, dd))
	except:
		# print "initialize"
		dd_all = dd

dif = np.vstack((sd_all, dd_all))
sim = np.hstack((np.ones(sd_all.shape[0]),np.zeros(dd_all.shape[0])))


print clf.predict(dif)
print clf.score(dif, sim)





log('Save model')
import cPickle as pickle
s = pickle.dump(clf, open('svm2','w'))
print pickle.load(open('svm2'))
'''
labeled/0664_02.jpg 937
labeled/0769_08.jpg 1301
labeled/0058_06.jpg 107
labeled/1376_09.jpg 1183
'''