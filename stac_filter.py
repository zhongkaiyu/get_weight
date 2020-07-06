import h5py
import keras
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
# from keras_applications.resnet import ResNet18
import resnet_un
import matplotlib.pyplot as plt


weights_file_path ='/home/yuzhongkai/resnet-imagenet/weights/imagenet_resnet18_3_026_mix.h5'

#加载model和weight文件-------------------------------------------------------
makeNetwork = lambda modelFunc : modelFunc(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model = makeNetwork(resnet_un.ResNet18)
model.load_weights(weights_file_path)
#----------------------------------------------------------------------------
# model = load_model(weights_file_path)
# model.summary()
#name_list可以用model.summary得到
name_list = ['conv0',
             'stage1_unit1_sc', 'stage1_unit1_conv1', 'stage1_unit1_conv2', 'stage1_unit2_conv1', 'stage1_unit2_conv2',
             'stage2_unit1_sc', 'stage2_unit1_conv1', 'stage2_unit1_conv2', 'stage2_unit2_conv1', 'stage1_unit2_conv2',
             'stage3_unit1_sc', 'stage3_unit1_conv1', 'stage3_unit1_conv2', 'stage3_unit2_conv1', 'stage1_unit2_conv2',
             'stage4_unit1_sc', 'stage4_unit1_conv1', 'stage4_unit1_conv2', 'stage4_unit2_conv1', 'stage1_unit2_conv2',
             ]
for layer in model.layers:
    weight = layer.get_weights()
    print(len(weight), layer.get_name())
