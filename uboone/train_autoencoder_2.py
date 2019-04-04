# Basic imports
import os,sys,time
import shutil,csv
from keras.layers import Input
from train import config

# Load configuration and check if it's good
cfg = config()
if not cfg.parse(sys.argv) or not cfg.sanity_check():
  sys.exit(1)

# Get the start iter number
start_iter=0
if cfg.LOAD_FILE:
  start_iter=int(cfg.LOAD_FILE.split('-')[1])

# Print configuration
print '\033[95mConfiguration\033[00m'
print cfg
time.sleep(0.5)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7,8"

# Import more libraries (after configuration is validated)
import numpy as np
import tensorflow as tf
import numpy as np
from dataloader import larcv_data

#
# Utility functions
#
# Integer rounder
def time_round(num,digits):
  return float( int(num * np.power(10,digits)) / float(np.power(10,digits)) )

#########################
# main part starts here #
#########################

#
# Step 0: configure IO
#

# Instantiate and configure
if not cfg.FILLER_CONFIG:
  'Must provide larcv data filler configuration file!'
  sys.exit(1)
proc = larcv_data()
filler_cfg = {'filler_name': 'DataFiller', 
              'verbosity':0, 
              'filler_cfg':cfg.FILLER_CONFIG}
proc.configure(filler_cfg)
# Spin IO thread first to read in a batch of image (this loads image dimension to the IO python interface)
proc.read_next(cfg.BATCH_SIZE)
# Force data to be read (calling next will sleep enough for the IO thread to finidh reading)
#print proc.next().shape
# Immediately start the thread for later IO

#print image_dim

from net import autoencoder
input_img = Input(shape=(512,512,1))

from keras.layers import Input, Dense, advanced_activations, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras import callbacks


AutoEncoder, Encoder=autoencoder.build(input_img, layers_num=5, drop_out=0.2, verbose=1)
#AutoEncoder.compile(optimizer='Adam', loss='mean_squared_error')

print AutoEncoder  

'''
mc = callbacks.ModelCheckpoint('trainweights/weights{epoch:08d}.h5',save_weights_only=True, period=1000)
AutoEncoder.fit(uboone_train, uboone_train,
                epochs=cfg.ITERATIONS,
                batch_size=5,
                shuffle=True,
                validation_data=(uboone_test, uboone_test),
                callbacks=[TensorBoard(log_dir='log/trainlog'), mc])
'''
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
import keras.backend as K

'''
def squareLoss(yTrue,yPred):
    diff=yTrue-yPred
    return K.sum(diff**2)
'''

def cosmic_squareLoss(y_true,y_pred):
  mask_value = 0
  #Cosmic related
  mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
  mask_true_sum = K.sum(mask_true)
  #Zero (background)
  mask_false = K.cast(K.equal(y_true, mask_value), K.floatx())
  mask_false_sum = K.sum(mask_false)
  
  mask_true_squared_error  = K.sum(K.square(mask_true  * (y_true - y_pred))) * (1 / mask_true_sum)
  mask_false_squared_error = K.sum(K.square(mask_false * (y_true - y_pred))) * (1 / mask_false_sum)
  #masked_mse = K.sum(masked_squared_error, axis = -1) / K.sum(mask_true, axis = -1)
  masked_squared_error = (mask_true_squared_error+mask_false_squared_error) * (1 / K.cast(K.shape(y_true)[0], K.floatx()))
  masked_mse = masked_squared_error
  return masked_mse

# This assumes that your machine has some available GPUs.
parallel_model  =  multi_gpu_model(AutoEncoder, gpus = 5)
#parallel_model.compile(optimizer = 'Adam', loss = 'binary_crossentropy')
parallel_model.compile(optimizer = 'Adam', loss = cosmic_squareLoss)
#parallel_model.compile(optimizer = 'Adam', loss = squareLoss)
#parallel_model.compile(optimizer = 'Adam', loss = 'mse')
#parallel_model.compile(optimizer='Adam', loss='mean_squared_error')


#num=cfg.BATCH_SIZE
def input_generator(num):
  while True:
    img=proc.next()
    proc.read_next(num)
    image_dim = proc.image_dim()
    img=img.reshape(num, image_dim[2], image_dim[3], 1)
    resize_img=np.zeros((num,512,512,1))
    idx=0
    for each in img:
      result=np.zeros((512,512,1))
      result[:,:,:1]=each[:512,:512,:1]
      resize_img[idx]=result
      idx+=1

    #yield [uboone_train,uboone_train], [uboone_test,uboone_test]
    yield (resize_img, resize_img)
'''
uboone_train=input_generator(cfg.BATCH_SIZE)
uboone_test=input_generator(cfg.BATCH_SIZE)

#uboone_train, uboone_test = input_generator(cfg.BATCH_SIZE).next()

mc = callbacks.ModelCheckpoint('trainweights/weights{epoch:08d}.h5',save_weights_only=True)

parallel_model.fit_generator(uboone_train,
                             epochs=cfg.ITERATIONS,
                             steps_per_epoch=10,
                             validation_data=uboone_test,
                             validation_steps=5,
                             callbacks=[TensorBoard(log_dir='log/trainlog'), mc])
'''

for step in xrange(cfg.ITERATIONS):
  print "=============>>>>>step %i"%step
  img=proc.next()
  proc.read_next(cfg.BATCH_SIZE)
  image_dim = proc.image_dim()
  print 'input image dimension, ',image_dim
  img=img.reshape(cfg.BATCH_SIZE, image_dim[2], image_dim[3], 1)
  resize_img=np.zeros((cfg.BATCH_SIZE,512,512,1))
  idx=0
  for each in img:
    result=np.zeros((512,512,1))
    result[:,:,:1]=each[:512,:512,:1]
    resize_img[idx]=result
    idx+=1

  input_img=np.zeros((cfg.BATCH_SIZE,512,512,1))
  idx=0

  for each in resize_img:
    each=each.reshape(512**2,)
    each=each/sum(each)
    input_img[idx]=each.reshape(512,512,1)
    idx+=1

  here=cfg.BATCH_SIZE/5
  uboone_test=resize_img[:here]
  uboone_train=resize_img[here:]

  print "training on %i events, val on %i events"%(uboone_train.shape[0], uboone_test.shape[0])
  print "input shape is ", uboone_train.shape
  print "max in image is ", np.max(uboone_train[0].flatten())
  
  

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  '''
  for x in xrange(9):
    fig,ax = plt.subplots(1,1,figsize=(20,18))
    ax.imshow(uboone_train[x].reshape(512,512))
    fig.savefig("foo_%i.pdf"%x)
  '''
  
  epochs=20
  parallel_model.fit(uboone_train, uboone_train,
                     epochs=(step+1)*epochs,
                     initial_epoch=step*epochs,
                     batch_size=20,
                     shuffle=True,
                     validation_data=(uboone_test, uboone_test),
                     callbacks=[TensorBoard(log_dir='log/trainlog')])
                     #callbacks=[TensorBoard(log_dir='log/trainlog', histogram_freq=1, write_images=1)])
  if(step % 10==0): 
    parallel_model.save_weights('trainweights/weights_%i.h5'%((step)))
    print "saving weights_%i.h5"%((step+1)*epochs)
print "Training Done"
