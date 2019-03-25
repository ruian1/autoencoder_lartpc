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
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# In[ ]:


input_img=Input(shape=(28,28,1))


# In[ ]:


x = Conv2D(16, (3, 3), activation=None, padding='same')(input_img)
x = PReLU(shared_axes=[1,2])(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print x
x = Conv2D(8, (3, 3), activation=None, padding='same')(x)
x = PReLU(shared_axes=[1,2])(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print x
x = Conv2D(8, (3, 3), activation= None, padding='same')(x)
x = PReLU(shared_axes=[1,2])(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
print encoded

# In[ ]:


encoder = Model(input_img, encoded)


# In[ ]:


x = Conv2D(8, (3, 3), activation=None, padding='same')(encoded)
x = PReLU(shared_axes=[1,2])(x)
x = UpSampling2D((2, 2))(x)
print x
x = Conv2D(8, (3, 3), activation=None, padding='same')(x)
x = PReLU(shared_axes=[1,2])(x)
x = UpSampling2D((2, 2))(x)
print x
x = Conv2D(16, (3, 3), activation=None)(x)
x = PReLU(shared_axes=[1,2])(x)
x = UpSampling2D((2, 2))(x)
print x
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

print decoded

# In[ ]:


autoencoder = Model(input_img, decoded)


# In[ ]:


