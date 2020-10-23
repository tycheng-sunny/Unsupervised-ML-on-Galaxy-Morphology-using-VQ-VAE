#=========================================================================#
# Extracting features from images using a pretrained VQ-vAE model         #
# Save the extracted features as FITS file for later use                  #
#=========================================================================#

import time
import csv
import os
import pickle
import h5py

import numpy as np
import sonnet as snt
import tensorflow as tf

from astropy.io import fits
from astropy.table import Table, Column
#------------------------------------------------------------------------#
import time
#timer starts
Tstart = time.time()

## VQ-VAE: only Encoder architecture ##
# residual nets (ResNets)
def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
    for i in range(num_residual_layers):
        h_i = tf.nn.relu(h)

        h_i = snt.Conv2D(
              output_channels=num_residual_hiddens,
              kernel_shape=(3, 3),
              stride=(1, 1),
              name="res3x3_%d" % i)(h_i)
        h_i = tf.nn.relu(h_i)

        h_i = snt.Conv2D(
              output_channels=num_hiddens,
              kernel_shape=(1, 1),
              stride=(1, 1),
              name="res1x1_%d" % i)(h_i)
        h += h_i
    return tf.nn.relu(h)

# encoder
class Encoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name='encoder'):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens / 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")(x)
        h = tf.nn.relu(h)

        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")(h)
        h = tf.nn.relu(h)

        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_3")(h)
        h = tf.nn.relu(h)

        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_4")(h)

        h = residual_stack(
            h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        return h

## FUNCTION for feature extraction ##
def vqvae_feature_extract(data_dict, savemodel_vqvae_fn):
    # Set hyper-parameters.
    image_size = 64

    # hyper-parameters for VQ-VAE
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    # This value is not that important, usually 64 works.
    # This will not change the capacity in the information-bottleneck.
    embedding_dim = 64

    # The higher this value, the higher the capacity in the information bottleneck.
    num_embeddings = 16

    # commitment_cost should be set appropriately. It's often useful to try a couple
    # of values. It mostly depends on the scale of the reconstruction cost
    # (log p(x|z)). So if the reconstruction cost is 100x higher, the
    # commitment_cost should also be multiplied with the same amount.
    commitment_cost = 0.25

    # Use EMA updates for the codebook (instead of the Adam optimizer).
    # This typically converges faster, and makes the model less dependent on choice
    # of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
    # developed afterwards). See Appendix of the paper for more details.
    vq_use_ema = False

    # This is only used for EMA updates.
    decay = 0.99

    # Build modules.
    tf.reset_default_graph()
    encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
    pre_vq_conv1 = snt.Conv2D(
                          output_channels=embedding_dim,
                          kernel_shape=(1, 1),
                          stride=(1, 1),
                          name="to_vq")

    if vq_use_ema:
        vq_vae = snt.nets.VectorQuantizerEMA(
                                         embedding_dim=embedding_dim,
                                         num_embeddings=num_embeddings,
                                         commitment_cost=commitment_cost,
                                         decay=decay)
    else:
        vq_vae = snt.nets.VectorQuantizer(
                                      embedding_dim=embedding_dim,
                                      num_embeddings=num_embeddings,
                                      commitment_cost=commitment_cost)

    # Process inputs with conv stack, finishing with 1x1 to get to correct size.
    x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 1))
    z = pre_vq_conv1(encoder(x))

    vq_output_train = vq_vae(z, is_training=False)
    vq_output_embeds = vq_vae.embeddings # the codebook ek [D,K]

    # Create TF session for vqvae
    saver = tf.train.Saver()
    sess = tf.train.SingularMonitoredSession()
    saver.restore(sess, savemodel_vqvae_fn) # load vqvae pre-trained model

    # retrieve the extracted features and restore it into an array
    N = 500 # run N each time
    runN = len(data_dict["id"])// N
    for i in range(runN):
        if i == 0:
            embeds_indice_4_imgs = sess.run(vq_output_train["encoding_indices"], feed_dict={x: data_dict['images'][i*N: (i+1)*N]})
        else:
            tmp = sess.run(vq_output_train["encoding_indices"], feed_dict={x: data_dict['images'][i*N: (i+1)*N]})
            embeds_indice_4_imgs = np.concatenate((embeds_indice_4_imgs, tmp), axis=0)

    tmp = sess.run(vq_output_train["encoding_indices"], feed_dict={x: data_dict['images'][runN* N:]})
    embeds_indice_4_imgs = np.concatenate((embeds_indice_4_imgs, tmp), axis=0)
    #print('Shape of input: ', np.shape(embeds_indice_4_imgs))

    # Flatten the array of features
    flatten_embeds = []
    for iN in range(len(embeds_indice_4_imgs)):
        flatten_embeds.append(embeds_indice_4_imgs[iN].flatten())
    flatten_embeds = np.transpose(np.array(flatten_embeds))
    #print(np.shape(flatten_embeds))
    #print(flatten_embeds)
    return flatten_embeds


## Data Preparation ##
# reading path
data_file = 'data.hdf5' # data
savemodel_path = 'pretrain_vqvae_model/' # if using a pretrained model: savemodel_path = 'pretrain_vqvae_model/'
savename = 'vqvae_pretrain_64k_ek16_1sigma_w01' #if using a pretrained model: savename = 'vqvae_pretrain_64k_ek16_1sigma_w01'
savemodel_vqvae_fn = savemodel_path + savename + '.ckpt'  # vqvae models saving path

## variables for data
image_size = 64
channel_size = 1

## read hdf5 file
def openhdf(filename):
    f = h5py.File(filename, 'r')
    return f

def reshape_flattened_image_batch(flat_image_batch):
    return flat_image_batch.reshape(-1, image_size, image_size, channel_size)  # convert to NHWC
def combine_batches(batch_list):
    images = np.vstack([reshape_flattened_image_batch(batch_list['img_1sigma'][:])])
    id = np.vstack([np.array(batch_list['OBJID'][:])]).reshape(-1,1)
    return {'images': images, 'id': id}

data_dict = combine_batches(openhdf(data_file))
data_dict['images'] = data_dict['images']
data_dict['id'] = data_dict['id']

## MAIN ##
# read the extracted features
flatten_embeds = vqvae_feature_extract(data_dict, savemodel_vqvae_fn) # flatten the extracted features of each image
iddata = data_dict['id'].reshape(-1) # objid

# customise the column names --> f1, f2, etc.
column_name = []
for i in range(len(flatten_embeds)):
    column_name.append('f'+str(i))
#print(column_name)

# save into a FITs file
for i in range(len(column_name)):
    if i == 0:
        t = Table([iddata, flatten_embeds[i]], names=('sdss_objid', column_name[i]))
    else:
        newcol = Column(flatten_embeds[i], name=column_name[i])
        t.add_column(newcol, index=i+1)
t.write(savename + '_extractedF.fits', format='fits', overwrite=True)

#timer
print('\n', '## CODE RUNTIME:', time.time()-Tstart) #Timer end
