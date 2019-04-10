import os, sys, gc, shutil
import pandas as pd
import ROOT
from larcv import larcv
import numpy as np
import tensorflow as tf

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)
sys.path.insert(0,os.path.join(BASE_PATH,"../.."))

from lib.config import config_loader
from lib.rootdata_autoencoder import ROOTData

#for Autoencoder
#ROOT_DIR = os.path.abspath("../../../")
ROOT_DIR = os.path.abspath("/usr/local/share/dllee_unified/autoencoder_lartpc")
sys.path.append(ROOT_DIR)  #To find local version of the library

from net import autoencoder
from keras.layers import Input, Dense, advanced_activations, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras import callbacks

input_img = Input(shape=(512,512,1))
AutoEncoder, Encoder=autoencoder.build(input_img, layers_num=5, drop_out=0.0, verbose=1)

from keras.callbacks import TensorBoard
import keras.backend as K

#BASE_PATH = os.path.realpath(__file__)
#BASE_PATH = os.path.dirname(BASE_PATH)
#sys.path.insert(0,BASE_PATH)
#sys.path.insert(0,os.path.join(BASE_PATH,".."))

larcv.LArbysLoader()

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def image_modify (img, cfg):
    img_arr = np.array(img.as_vector())
    img_arr = np.where(img_arr<cfg.adc_lo,         0,img_arr)
    img_arr = np.where(img_arr>cfg.adc_hi,cfg.adc_hi,img_arr)
    img_arr = img_arr/cfg.adc_hi
    img_arr = img_arr.reshape(cfg.xdim,cfg.ydim, 1).astype(np.float32)
    
    return img_arr

def calc_loss(img1, img2):
    img1=img1.reshape(512**2,)
    img2=img2.reshape(512**2,)
    
    diff=img1-img2
    
    pos_loc=img1>0.0
    neg_loc=img1<=0.0


    pos_sum=0
    neg_sum=0

    if(sum(pos_loc)):
        pos_sum=np.sum(pos_loc*((img1-img2)**2))*(1./np.sum(pos_loc))
    if(sum(neg_loc)):
        neg_sum=np.sum(neg_loc*((img1-img2)**2))*(1./np.sum(neg_loc))
    
    loss=pos_sum + neg_sum
    
    return loss

def main(IMAGE_FILE,VTX_FILE,OUT_DIR,CFG):

    class_names=[0, 11, -11, 13, -13, 22, 211, -211, 2212]

    cfg  = config_loader(CFG)
    assert cfg.batch == 1

    plane=2
    weight_file = ""
    exec("weight_file = cfg.weight_file_autoencoder_plane%d" % plane)
    AutoEncoder.load_weights(weight_file)

    rd = ROOTData()

    NUM = int(os.path.basename(VTX_FILE).split(".")[0].split("_")[-1])
    FOUT = os.path.join(OUT_DIR,"autoencoder_out_%04d.root" % NUM)
    tfile = ROOT.TFile.Open(FOUT,"RECREATE")
    tfile.cd()
    #print "OPEN %s"%FOUT

    tree  = ROOT.TTree("autoencoder_tree","")
    rd.init_tree(tree)
    rd.reset()

    #
    #initialize iomanager
    #

    iom  = larcv.IOManager(larcv.IOManager.kREAD)
    iom.add_in_file(IMAGE_FILE)
    iom.add_in_file(VTX_FILE)
    iom.initialize()

    for entry in xrange(iom.get_n_entries()):

        if entry > 10: continue
        
        iom.read_entry(entry)

        ev_pgr = iom.get_data(larcv.kProductPGraph,"inter_par")
        ev_par = iom.get_data(larcv.kProductPixel2D,"inter_par_pixel")
        ev_pix = iom.get_data(larcv.kProductPixel2D,"inter_img_pixel")
        ev_int = iom.get_data(larcv.kProductPixel2D,"inter_int_pixel")
        #ev_img = iom.get_data(larcv.kProductImage2D,"wire")
        
        
        rd.run[0]    = int(ev_pix.run())
        rd.subrun[0] = int(ev_pix.subrun())
        rd.event[0]  = int(ev_pix.event())
        rd.entry[0]  = int(iom.current_entry())

        rd.num_vertex[0] = int(ev_pgr.PGraphArray().size())

        for ix,pgraph in enumerate(ev_pgr.PGraphArray()):
            #print "@pgid=%d" % ix
            #if (ix != 2): continue
            rd.vtxid[0] = int(ix)
   
            pgr = ev_pgr.PGraphArray().at(ix)
            cindex_v = np.array(pgr.ClusterIndexArray())
            
            #pixel2d_par_vv = ev_par.Pixel2DClusterArray()
            pixel2d_pix_vv = ev_pix.Pixel2DClusterArray()
            pixel2d_int_vv = ev_int.Pixel2DClusterArray()

            #parid = pgraph.ClusterIndexArray().front()
            #roi0 = pgraph.ParticleArray().front()

            #x = roi0.X()
            #y = roi0.Y()
            #z = roi0.Z()

            #y_2d_plane_0 = ROOT.Double()

            for plane in xrange(3):
                if plane!=2 : continue
                #print "@plane=%d" % plane
                
                ###Get 2D vertex Image
                
                #meta = roi0.BB(plane)

                #x_2d = ROOT.Double()
                #y_2d = ROOT.Double()
                
                #whole_img = ev_img.at(plane)
                
                #larcv.Project3D(whole_img.meta(), x, y, z, 0.0, plane, x_2d, y_2d)
                ##print 'x2d, ', x_2d, 'y2d, ',y_2d
                
                #if (plane == 0) : y_2d_plane_0 = y_2d
                #else : y_2d = y_2d_plane_0
                
                ###
                '''
                weight_file = ""
                exec("weight_file = cfg.weight_file_autoencoder_plane%d" % plane)
                AutoEncoder.load_weights(weight_file)
                '''
                #nothing
                #if pixel2d_vv.empty()==True: continue

                pixel2d_pix_v = pixel2d_pix_vv.at(plane)
                pixel2d_pix = pixel2d_pix_v.at(ix)

		pixel2d_int_v = pixel2d_int_vv.at(plane)
                pixel2d_int = pixel2d_int_v.at(ix)
            
                #nothing on this plane
                #if pixel2d_pix.empty() == True: continue

                img_pix = larcv.cluster_to_image2d(pixel2d_pix,cfg.xdim,cfg.ydim)
                img_int = larcv.cluster_to_image2d(pixel2d_int,cfg.xdim,cfg.ydim)

                img_pix_arr = image_modify(img_pix, cfg)
                img_int_arr = image_modify(img_int, cfg)

                #Detection
                output_pix=AutoEncoder.predict(img_pix_arr.reshape(1,512,512,1))
                output_int=AutoEncoder.predict(img_int_arr.reshape(1,512,512,1))

                rd.inferred[0] = 1


                
                print '========================>>>>>>>>>>>>>>>>>>>>'
                print 'run, subrun, event, ix',ev_pix.run(),ev_pix.subrun(),ev_pix.event(), ix


                print 'max is ', np.max(img_pix_arr)

                pix_loss=calc_loss(img_pix_arr, output_pix)
                int_loss=calc_loss(img_int_arr, output_int)
                
                print 'pix loss', pix_loss
                print 'int loss', int_loss


                import matplotlib.pyplot as plt
                fig,((ax1, ax2),(ax3, ax4))=plt.subplots(2,2,figsize=(16,10))
                ax1.imshow(img_pix_arr[:,:,0])
                ax1.set_title("pix_%i_%i_%i_%i_loss_%f"%(ev_pix.run(),ev_pix.subrun(),ev_pix.event(),ix, pix_loss))
                ax2.imshow(output_pix.reshape(512,512))
                ax3.imshow(img_int_arr[:,:,0])
                ax1.set_title("int_%i_%i_%i_%i_loss_%f"%(ev_pix.run(),ev_pix.subrun(),ev_pix.event(),ix, int_loss))
                ax4.imshow(output_int.reshape(512,512))
                fig.savefig("%i_%i_%i_%i"%(ev_pix.run(),ev_pix.subrun(),ev_pix.event(),ix), bbox_inches='tight')

                
                rd.autoencoder_score_pix[0]=pix_loss
                rd.autoencoder_score_int[0]=int_loss
                
            tree.Fill()
            rd.reset_vertex()
    tfile.cd()
    tree.Write()
    tfile.Close()
    iom.finalize()

if __name__ == '__main__':
    
    if len(sys.argv) != 5:
        print
        print "\tIMAGE_FILE = str(sys.argv[1])"
        print "\tVTX_FILE   = str(sys.argv[2])"
        print "\tOUT_DIR    = str(sys.argv[3])"
        print "\tCFG        = str(sys.argv[4])"
        print 
        sys.exit(1)
    
    IMAGE_FILE = str(sys.argv[1]) 
    VTX_FILE   = str(sys.argv[2])
    OUT_DIR    = str(sys.argv[3])
    CFG        = str(sys.argv[4])

    #CFG = os.path.join(BASE_PATH,"cfg","simple_config.cfg")

    with tf.device('/cpu:0'):
        main(IMAGE_FILE,VTX_FILE,OUT_DIR,CFG)
    
    print "DONE!"
    sys.exit(0)
