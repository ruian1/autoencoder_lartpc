# AutoEncoder_LArTPC(Keras): AutoEncoder for LArTPC, Keras version

AutoEncoder applicaiton for LArTPC. Network taks a 512x512 LArTPC images and returns score value for weather it contains a neutrion of just cosmics.

# Dependecies:
[LArCV](https://github.com/LArbys/LArCV),
ROOT,
Keras

# Setup:
0. Setup LArCV using [Wiki](https://github.com/LArbys/LArCV),
1. git clone https://github.com/ruian1/autoencoder_lartpc.git
2. source setup.sh

# Training:
0. Edit training configures under ./uboone/
1. python ./uboone/train_autoencoder_2.py	./uboone/autoencoder_2.cfg	

# Inference:
0. cd ./uboone/production/
1. python inference_mrcnn_center.py image2d.root (for [image2d](https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/Image2D.h) input).
2. python inference_autoencoder.py	pixel2d.root (for [pixel2d](https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/Pixel2D.h) input).
