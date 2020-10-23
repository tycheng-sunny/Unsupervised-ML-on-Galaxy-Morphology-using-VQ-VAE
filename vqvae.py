#=========================================================================#
# Feature learning by a Vector-Quantised Variational Autoencoder (VQ-VAE) #
#=========================================================================#

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0" # for GPU
#import matplotlib as mpl
#mpl.use('Agg') # to use matplotlib without visualisation envs
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf

from sklearn_extra.cluster import KMedoids
from sklearn import metrics
from sklearn.utils import shuffle
from six.moves import xrange
#------------------------------------------------------------------------#
import time
#timer starts
Tstart = time.time()

## VQ-VAE: Encoder & Decoder architecture ##
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

# decoder
class Decoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name='decoder'):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1")(x)

        h = residual_stack(
            h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

        h = snt.Conv2DTranspose(
            output_channels=int(self._num_hiddens),
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")(h)
        h = tf.nn.relu(h)

        h = snt.Conv2DTranspose(
            output_channels=int(self._num_hiddens / 2),
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3")(h)
        h = tf.nn.relu(h)

        x_recon = snt.Conv2DTranspose(
                  output_channels=1,
                  output_shape=None,
                  kernel_shape=(4, 4),
                  stride=(2, 2),
                  name="dec_4")(h)

        return x_recon

#=================================================================#
## Hyper-parameters for the VQ-VAE ##
# Train mode
Train = True # True: training a new model and save the model. False: reloading the pretrained model
# Set hyper-parameters.
image_size = 64
num_feature = 64 # number of feature extracted from the VQ-VAE
batch_size = 32 # When Train=False, batch_size=4 which means output 4 galaxy images

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU (guideline from Sonnet library).
min_training_epoch = 5000 # minimum training epoches without saving models # default could try 20k when you have time
num_training_updates = min_training_epoch + 1000 # extra epoches, save models when local minimum is reached # default could try 10k when you have time

# These hyper-parameters define the size of the model (encoder and decoder; number of parameters and layers).
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

# This value is not that important, usually 64 works (guideline from Sonnet library).
# This will not change the capacity in the information-bottleneck.
embedding_dim = 64

# codebook size: the higher this value, the higher the capacity in the information bottleneck.
num_embeddings = 16

# (guideline from Sonnet library)
# commitment_cost should be set appropriately. It's often useful to try a couple
# of values. It mostly depends on the scale of the reconstruction cost
# (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.
commitment_cost = 0.25

# (guideline from Sonnet library)
# Use EMA updates for the codebook (instead of the Adam optimizer).
# This typically converges faster, and makes the model less dependent on choice
# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
vq_use_ema = False

# (guideline from Sonnet library)
# This is only used for EMA updates.
decay = 0.99

learning_rate = 3e-4

#=================================================================#
## Data Preparation ##
# reading path
data_file = 'data.hdf5' # data
savemodel_path = 'vqvae_model_output/' # if Train=False, using a pretrained model: savemodel_path = 'pretrain_vqvae_model/'
savename = 'vqvae_ek16_1sigma_w01' # any filename you would like # if Train=False, using a pretrained model: savename = 'vqvae_pretrain_64k_ek16_1sigma_w01'
savemodel_vqvae_fn = savemodel_path + savename  # vqvae models saving path

# hyper-parameters for retrieving the data
ori_image_size = 64
channel_size = 1

# read hdf5 file
def openhdf(filename):
    f = h5py.File(filename, 'r')
    return f
def reshape_flattened_image_batch(flat_image_batch):
    return flat_image_batch.reshape(-1, ori_image_size, ori_image_size, channel_size)  # convert to NHWC
def combine_batches(batch_list):
    images = np.vstack([reshape_flattened_image_batch(batch_list['img_1sigma'][:])])
    id = np.vstack([np.array(batch_list['OBJID'][:])]).reshape(-1,1)
    return {'images': images, 'id': id}
def random_shuffle(dict):
    dict["images"], dict["id"] = shuffle(dict["images"], dict["id"], random_state=0)
    return dict

data_dict = random_shuffle(combine_batches(openhdf(data_file)))
train_data_num = np.int(0.90* len(data_dict["id"])) # the number of training data # can change the ratio (default=0.9)
print('Number of training dataset:', train_data_num)
print('Number of validation dataset:', (len(data_dict["id"]) - train_data_num))

data_variance = np.var(data_dict["images"][:train_data_num]) # for the normalisation of the reconstruction loss (mse)

#=================================================================#
## MAIN training code ##
# data loading function
def decode_and_get_image(x):
    data = np.array(data_dict["images"][np.where(data_dict["id"] == x)[0][0]])
    return data

def load(x):
    x = x.numpy()
    x = np.array(list(map(decode_and_get_image, x)))
    return x

def loader(y):
    imgs = tf.py_function(load, [y], tf.float32)
    imgs = tf.cast(imgs, tf.float32)
    return imgs[0]

# load training data
tf.reset_default_graph()
train_paths = tf.data.Dataset.from_tensor_slices(data_dict["id"][:train_data_num]) # load the id instead of images
train_dset = train_paths.map(loader)

train_dset = train_dset.repeat(-1).shuffle(10000).batch(batch_size)
train_iterator = train_dset.make_one_shot_iterator()
train_dataset_batch = train_iterator.get_next()

# load validation data
valid_paths = tf.data.Dataset.from_tensor_slices(data_dict["id"][train_data_num:])
valid_dset = valid_paths.map(loader)

valid_dset = valid_dset.repeat(-1).batch(batch_size)
valid_iterator = valid_dset.make_one_shot_iterator()
valid_dataset_batch = valid_iterator.get_next()

def get_images(sess, subset='train'):
    if subset == 'train':
        return sess.run(train_dataset_batch)
    elif subset == 'valid':
        return sess.run(valid_dataset_batch)

# Build modules
encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)
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
s = tf.placeholder(tf.float32, shape=()) #silhouette score
z = pre_vq_conv1(encoder(x)) #latent space

# vq_output_train["quantize"] are the quantized outputs of the encoder.
# That is also what is used during training with the straight-through estimator.
# To get the one-hot coded assignments use vq_output_train["encodings"] instead.
# These encodings will not pass gradients into to encoder,
# but can be used to train a PixelCNN on top afterwards.

# For training (VQ-VAE)
vq_output_train = vq_vae(z, is_training=True)
extracted_features = vq_output_train["encoding_indices"]
x_recon = decoder(vq_output_train["quantize"])

# For training - optimisation factor
recon_error = tf.reduce_mean((x_recon - x)**2) / data_variance  # Normalized MSE # reconstruction loss
score_penalty = (1.-s)* 0.1 # newly defined penalty for clustering performance # weight used in our work is 0.1
loss = recon_error + vq_output_train["loss"] + score_penalty #total loss: reconstructed loss + commitment loss + codebook loss + score penalty

# For evaluation, make sure is_training=False!
vq_output_eval = vq_vae(z, is_training=False)
x_recon_eval = decoder(vq_output_eval["quantize"])

# The following is a useful value to track during training.
# It indicates how many codes are 'active' on average.
perplexity = vq_output_train["perplexity"]

# Create optimizer and TF session.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)
saver = tf.train.Saver()
sess = tf.train.SingularMonitoredSession()

# Special adapted code for saving model when using MonitoredSession() # Don't know why yet...
def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
    return session

def flatten_imgarray(input):
    # flatten the array of extracted features for clustering
    flatten_embeds = []
    for i in range(len(input)):
        flatten_embeds.append(input[i].flatten())
    flatten_embeds = np.array(flatten_embeds)
    #print(np.shape(flatten_embeds))
    return flatten_embeds

# set the Train mode. If Train=True: do training and save the model, Train=False: reloading the pre-trained model
if Train:
    # Train.
    train_loss, train_loss_print, train_loss_zoom = [], [], [] # record the total loss
    train_res_recon_error, train_rre_print, train_rre_zoom = [], [], [] # record the reconstruction loss
    train_score, train_score_print, train_score_zoom = [], [], [] # record the score_penalty loss
    train_res_perplexity, train_perplexity_print, train_perplexity_zoom = [], [], [] # record the perplexity (how many vectors in the codebook are used)

    valid_res_recon_error, valid_rre_print, valid_rre_zoom = [], [], []
    for i in xrange(num_training_updates):
        if (i+1) < 1000:
            # first 1000 epochs don't do clustering
            feed_dict = {x: get_images(sess), s:1.}
            results = sess.run([train_op, recon_error, perplexity, loss, extracted_features], feed_dict=feed_dict)

        else:
            # clustering to obtain two clusters for measuring sillhoute score # distance metric: hamming distance
            flatten_embeds = flatten_imgarray(results[4])
            kmedoids_model = KMedoids(n_clusters=2, metric='hamming', init='k-medoids++').fit(flatten_embeds)
            silhouette_score = metrics.silhouette_score(flatten_embeds, kmedoids_model.labels_, metric='hamming')
            feed_dict = {x: get_images(sess), s: silhouette_score}
            results = sess.run([train_op, recon_error, perplexity, loss, extracted_features], feed_dict=feed_dict)

            #train record
            train_res_recon_error.append(results[1])
            train_res_perplexity.append(results[2])
            train_loss.append(results[3])
            train_score.append(silhouette_score)

            #print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))

            #validation record
            valid_results = sess.run([recon_error], feed_dict={x: get_images(sess, subset='valid'), s: silhouette_score})
            valid_res_recon_error.append(valid_results[0])

            if (i+1) % 100 == 0:
                # print out training information # per 100 epoches
                print('%d iterations' % (i+1))
                print('train_loss: %.3f' % np.mean(train_loss[-100:]))
                print('train_recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('train_silhouette_score: %.2f' %np.mean(train_score[-100:]))
                train_rre_print.append(np.mean(train_res_recon_error[-100:]))
                train_loss_print.append(np.mean(train_loss[-100:]))
                train_perplexity_print.append(np.mean(train_res_perplexity[-100:]))
                train_score_print.append(np.mean(train_score[-100:]))

                #validation
                print('valid_recon_error: %.3f' % np.mean(valid_res_recon_error[-100:]))
                valid_rre_print.append(np.mean(valid_res_recon_error[-100:]))

            if (i+1) >= min_training_epoch:
                # when training epoches >= min_training_epoch, start to save vqvae models when the local minimum is reached
                print('%d iterations' % (i+1))
                print('train_loss: %.3f' % train_loss[-1:][0])
                print('train_recon_error: %.3f' % train_res_recon_error[-1:][0])
                print('train_silhouette_score: %.2f' %train_score[-1:][0])
                train_rre_zoom.append(train_res_recon_error[-1:][0])
                train_loss_zoom.append(train_loss[-1:][0])
                train_perplexity_zoom.append(train_res_perplexity[-1:][0])
                train_score_zoom.append(train_score[-1:][0])
                #print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))

                #validation
                print('valid_recon_error: %.3f' % valid_res_recon_error[-1:][0])
                valid_rre_zoom.append(valid_res_recon_error[-1:][0])

                if (i+1) == min_training_epoch:
                    # initial loss as the reference
                    local_max = train_score[-1:][0] # sillhoute score
                    train_loss_avg = np.mean(train_loss[-100:]) # average of traing loss in 100 epoches
                    val_rre_min = valid_res_recon_error[-1:][0] # reconstructed loss

                else:
                    # sillhoute score - larger value is better
                    # total train loss needs to be at least large than average in 100 epoches
                    # reconstructed loss needs to be local minimum
                    local_tmp = train_score[-1:][0]
                    train_loss_tmp = train_loss[-1:][0]
                    val_rre_tmp = valid_res_recon_error[-1:][0]
                    if local_tmp >= local_max and train_loss_tmp <= train_loss_avg and val_rre_tmp <= val_rre_min:
                        local_max = local_tmp
                        train_loss_avg = np.mean(train_loss[-100:])
                        val_rre_min = val_rre_tmp
                        saver.save(get_session(sess), savemodel_vqvae_fn + '_ep' + str(i+1) + '_s' + str(np.int(np.round(train_score[-1:][0],3)*1000)) + '.ckpt')

    # Output reconstruction loss and average codebook usage
    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,3,1)
    ax.plot(train_rre_print, label='train_recon_err', alpha=0.5)
    ax.plot(valid_rre_print, label='valid_recon_err', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('epochs(*100)+1k')
    ax.set_title('Average NMSE.')
    plt.legend()

    ax = f.add_subplot(1,3,2)
    ax.plot(train_score_print, label='train_silhouette_score', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('epochs(*100)+1k')
    ax.set_title('Average silhouette score')
    plt.legend()

    ax = f.add_subplot(1,3,3)
    ax.plot(train_perplexity_print)
    ax.set_title('Average codebook usage (perplexity).')
    ax.set_xlabel('epochs(*100)+1k')
    plt.savefig('loss_' + savename + '.png')

    # (Zoom) Output reconstruction loss and average codebook usage after min_training_epoch
    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,3,1)
    ax.plot(train_rre_zoom, label='train_recon_err', alpha=0.5)
    ax.plot(valid_rre_zoom, label='valid_recon_err', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('epochs(+51k)')
    ax.set_title('NMSE.')
    plt.legend()

    ax = f.add_subplot(1,3,2)
    ax.plot(train_score_zoom, label='train_silhouette_score', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('epochs(+51k)')
    ax.set_title('silhouette score')
    plt.legend()

    ax = f.add_subplot(1,3,3)
    ax.plot(train_perplexity_zoom)
    ax.set_title('Average codebook usage (perplexity).')
    ax.set_xlabel('epochs(+51k)')
    plt.savefig('loss_' + savename + '_zoom.png')

else:
    saver.restore(sess, savemodel_vqvae_fn + '.ckpt')

    # Reconstructions
    #sess.run(valid_dataset_iterator.initializer)
    train_originals = get_images(sess, subset='train')
    train_reconstructions = sess.run(x_recon_eval, feed_dict={x: train_originals})
    valid_originals = get_images(sess, subset='valid')
    valid_reconstructions = sess.run(x_recon_eval, feed_dict={x: valid_originals})

    def convert_batch_to_image_grid(image_batch):
        reshaped = (image_batch.reshape(2, 2, image_size, image_size) # batch_size and image_size
                .transpose(0, 2, 1, 3)
                .reshape(2 * image_size, 2 * image_size))
        return reshaped + 0.5

    # Plot the comparison between original, reconstruction, and residual images
    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(2,3,1)
    ax.imshow(convert_batch_to_image_grid(train_originals),
          interpolation='nearest', cmap='gray_r')
    ax.set_title('training data originals')
    plt.axis('off')

    ax = f.add_subplot(2,3,2)
    ax.imshow(convert_batch_to_image_grid(train_reconstructions),
          interpolation='nearest', cmap='gray_r')
    ax.set_title('training data reconstructions')
    plt.axis('off')

    ax = f.add_subplot(2,3,3)
    ax.imshow(convert_batch_to_image_grid(train_originals) - convert_batch_to_image_grid(train_reconstructions), interpolation='nearest', cmap='gray_r')
    ax.set_title('training data residual')
    plt.axis('off')

    ax = f.add_subplot(2,3,4)
    ax.imshow(convert_batch_to_image_grid(valid_originals),
          interpolation='nearest', cmap='gray_r')
    ax.set_title('validation data originals')
    plt.axis('off')

    ax = f.add_subplot(2,3,5)
    ax.imshow(convert_batch_to_image_grid(valid_reconstructions),
          interpolation='nearest', cmap='gray_r')
    ax.set_title('validation data reconstructions')
    plt.axis('off')

    ax = f.add_subplot(2,3,6)
    ax.imshow(convert_batch_to_image_grid(valid_originals) - convert_batch_to_image_grid(valid_reconstructions), interpolation='nearest', cmap='gray_r')
    ax.set_title('validation data residual')
    plt.axis('off')

    plt.savefig('reconstruction_' + savename + '.png')

#timer
print('\n', '## CODE RUNTIME:', time.time()-Tstart) #Timer end
