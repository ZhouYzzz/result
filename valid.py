#!/usr/bin/python
import numpy as np
import sys
sys.path.insert(0, '/home/gpu/zhouyz/caffe/python')
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(sys.argv[1], caffe.TEST)