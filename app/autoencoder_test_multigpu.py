#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers import Input, Dense, advanced_activations, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras import callbacks


# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

# In[ ]:


input_img=Input(shape=(28,28,1))


# In[ ]:


x = Conv2D(16, (3, 3), activation=None, padding='same')(input_img)
x = PReLU(shared_axes=[1,2])(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation=None, padding='same')(x)
x = PReLU(shared_axes=[1,2])(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation= None, padding='same')(x)
x = PReLU(shared_axes=[1,2])(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


# In[ ]:


encoder = Model(input_img, encoded)


# In[ ]:


x = Conv2D(8, (3, 3), activation=None, padding='same')(encoded)
x = PReLU(shared_axes=[1,2])(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation=None, padding='same')(x)
x = PReLU(shared_axes=[1,2])(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation=None)(x)
x = PReLU(shared_axes=[1,2])(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)

# In[ ]:

from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
# In[ ]:

parallel_model = multi_gpu_model(autoencoder, gpus=3)

parallel_model.compile(optimizer='Adam', loss='binary_crossentropy')
#parallel_model.compile(optimizer='Adam', loss='mean_squared_error')

#autoencoder.compile(optimizer='Adam', loss='mean_squared_error')
#autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')


# In[ ]:


from keras.datasets import mnist
import numpy as np

(x_train, train_label), (x_test, test_label) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 


x2_train=x_train[np.where(train_label==2)]
x2_test =x_test[np.where(test_label==2)]


epochs=50000
period=99


# model.fit(X_train, Y_train, callbacks=[mc])


# In[ ]:


from keras.callbacks import TensorBoard

print x2_train.shape

sliced=int(x2_train.shape[0]/5)

mc = callbacks.ModelCheckpoint('log/test/weights{epoch:08d}.h5',save_weights_only=True, period=period)

for x in xrange(5):
    print "Trainng on %i/5 data."%(x+1)
    
    x2_train_slice=x2_train[sliced*x:sliced*(x+1)]
    
    print "start %i, end %i"%(sliced*x, sliced*(x+1))
    
    print "training on %i events."%x2_train_slice.shape[0]
    
    parallel_model.fit(x2_train_slice, x2_train_slice,
    #autoencoder.fit(x2_train_slice, x2_train_slice,
                       epochs=100+x*100,
                       batch_size=500,
                       initial_epoch=x*100,
                       shuffle=True,
                       validation_data=(x2_test, x2_test),
                       callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    parallel_model.save_weights('log/test/weights_%i.h5'%((x+1)*100))
