from keras.layers import Input, Dense, advanced_activations, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras import callbacks
from keras.regularizers import l1

def build(input_img, verbose=0, layers_num=5, drop_out=0.0):

#input_img=Input(shape=(28,28,1))

   down_layers=layers_num
   up_layers=layers_num

   x=input_img
   if(verbose): print "input, ", x
   for down_layer in xrange(down_layers):
      filter_num=down_layer+1
      if (verbose): print filter_num
      '''
      x = Conv2D(32*(2**filter_num), (3, 3), activation=None, padding='same', activity_regularizer=l1(1e-5))(x)
      x = BatchNormalization()(x)
      x = PReLU(shared_axes=[1,2])(x)
      x = Dropout(drop_out)(x)
      x = MaxPooling2D((2, 2), padding='same')(x)
      '''
      x = Conv2D(32*(2**filter_num), (3, 3), activation='relu', padding='same')(x)
      x = BatchNormalization()(x)
      x = MaxPooling2D((2, 2), padding='same')(x)

      if (verbose): print x

   encoder = Model(input_img, x)

   for up_layer in xrange(up_layers):
      filter_num=layers_num-up_layer-1
      if (verbose): print filter_num
      '''
      x = Conv2D(32*(2**filter_num), (3, 3), activation=None, padding='same', activity_regularizer=l1(1e-5))(x)
      x = BatchNormalization()(x)
      x = PReLU(shared_axes=[1,2])(x)
      x = Dropout(drop_out)(x)
      x = UpSampling2D((2, 2))(x)
      '''
      x = Conv2D(32*(2**filter_num), (3, 3), activation='relu', padding='same')(x)
      x = BatchNormalization()(x)
      x = UpSampling2D((2, 2))(x)
      

      if (verbose): print x

   #decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', activity_regularizer=l1(1e-5))(x)
   decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
   if (verbose): print "decoded",decoded
   autoencoder = Model(input_img, decoded)

   return autoencoder, encoder


if __name__ == '__main__':
   import os
   os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
   os.environ["CUDA_VISIBLE_DEVICES"]="8"
   x = Input(shape=(512,512,1))
   net = build(x, verbose=1, layers_num=5)
    
