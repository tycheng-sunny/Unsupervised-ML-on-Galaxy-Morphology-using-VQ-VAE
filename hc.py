#===========================================================================================#
# Using Hierarchical Clustering to categorise images represented by a set of extracted      #
# features in high-dimensional feature space.                                               #
# - Set "Train=False" first to get the merger tree in HC                                    #
# - Then set the number of clusters needed and "Train=True" to train HC                     #
#===========================================================================================#
import csv
import os
import h5py

import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import ward, fcluster, dendrogram
from scipy.spatial.distance import pdist
from astropy.io import fits
from sklearn.utils import shuffle

#------------------------------------------------------------------------#
import time
#timer starts
Tstart = time.time()

## FUNCTION ##
def readtxt(list):
    open_list = open(list, 'r') #read file
    lineOFlist = open_list.readlines() #read every lines in file

    #read item in each column per row, and store it as a list
    distance = [np.round(np.float(row[0:1][0][:]), 8) for row in csv.reader(lineOFlist[:],delimiter=' ',skipinitialspace = "true")]
    open_list.close()
    return distance

def readcata_genarray(filename):
    f = fits.open(filename)
    tab = f[1].data
    cols = f[1].columns
    tab_array = []
    for i in cols.names:
        if not i == 'sdss_objid':
            tab[i] = shuffle(tab[i], random_state=0)
            tab_array.append(tab[i])
    output_arr = np.array(tab_array).T
    print(np.shape(output_arr))
    return output_arr

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        dist_ = []
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points', va='top', ha='center')
                dist_.append(y)
    if max_d:
        plt.axhline(y=max_d, c='k')
    return ddata, dist_

#------------------------------------------------------------------------#
## Data Preparation ##
# reading path
data_ori = 'data.hdf5' # data
savename_vqvae = 'vqvae_pretrain_64k_ek16_1sigma_w01' # if using a pretrained model: savename_vqvae = 'vqvae_pretrain_64k_ek16_1sigma_w01'
data_file = savename_vqvae + '_extractedF.fits' # extracted features
savename_hc = 'hc_' + savename_vqvae

# hyper-parameters for data
image_size = 64
channel_size = 1

# read hdf5 file
# for showing original images after grouping
def openhdf(filename):
    f = h5py.File(filename, 'r')
    return f

def reshape_flattened_image_batch(flat_image_batch):
    return flat_image_batch.reshape(-1, image_size, image_size, channel_size)  # convert to NHWC

def combine_batches(batch_list):
    orimgs = np.vstack([reshape_flattened_image_batch(batch_list['img_raw'][:])])
    id = np.vstack([np.array(batch_list['OBJID'][:])]).reshape(-1,1)
    return {'orimgs': orimgs, 'id': id}

def random_shuffle(dict):
    dict["orimgs"], dict["id"] = shuffle(dict["orimgs"], dict["id"], random_state=0)
    return dict

data_dict = random_shuffle(combine_batches(openhdf(data_ori)))

# input for hc
flatten_embeds = readcata_genarray(data_file)
print(np.shape(flatten_embeds))

#------------------------------------------------------------------------#
## MAIN ##
Train = True #False: plot merger tree diagram; True: training HC
Ncluster = 2 # number of clusters wanted

if Train:
    # Hierarchical Clustering
    Z = ward(pdist(flatten_embeds, metric='hamming')) # measure the distances between images
    distance = readtxt('hc_' + savename_vqvae + 'merge_node.txt')
    distance = np.flip(distance, 0) #sort down

    dist = distance[Ncluster-2]
    grouper = fcluster(Z, t=dist-0.00001, criterion='distance')
    print('Number of cluster %d' %np.max(grouper))

    # categorise each image
    y_pred = grouper
    y_max = np.max(y_pred)
    group = [ [] for _ in range(y_max)]
    id_group = [ [] for _ in range(y_max)]
    for ix in range(len(y_pred)):
        for ig in range(len(group)):
            if y_pred[ix] == ig+1:
                group[ig].append(data_dict["orimgs"][ix].reshape(image_size, image_size))
                id_group[ig].append(data_dict["id"][ix][0])
            else:
                continue

    # plot examples of each group
    for n in range(len(group)):
        print('Number of data in group %d: ' %(n+1) + '%d' %len(id_group[n]))
        # Output 25 examples for each group unless the number of the group is less than 25, then output all of them.
        fig = plt.figure(figsize=(15,15))
        if len(id_group[n]) >= 25:
            ax1 = [None]* 25
            random_id = np.random.choice(id_group[n], 25, replace=False)
            for jshow in range(len(random_id)):
                loc = id_group[n].index(random_id[jshow])
                ax1[jshow] = fig.add_subplot(5,5,jshow+1)
                ax1[jshow].imshow(group[n][loc], cmap='gray_r')
                ax1[jshow].set_xticks([])
                ax1[jshow].set_xticklabels([])
                ax1[jshow].set_xlabel(str(random_id[jshow]))
                cur_axes = plt.gca()
                cur_axes.axes.get_yaxis().set_visible(False)
            plt.savefig(savename_hc + '_c' + str(Ncluster) + '_g' + str(n+1) + '.jpg')
        else:
            ax1 = [None]* len(id_group[n])
            for jshow in range(len(id_group[n])):
                loc = id_group[n].index(id_group[n][jshow])
                ax1[jshow] = fig.add_subplot(5,5,jshow+1)
                ax1[jshow].imshow(group[n][loc], cmap='gray_r')
                ax1[jshow].set_xticks([])
                ax1[jshow].set_xticklabels([])
                ax1[jshow].set_xlabel(str(random_id[jshow]))
                cur_axes = plt.gca()
                cur_axes.axes.get_yaxis().set_visible(False)
            plt.savefig(savename_hc + '_c' + str(Ncluster) + '_g' + str(n+1) + '.jpg')

else:
    Z = ward(pdist(flatten_embeds, metric='hamming'))

    # draw the merger tree
    plt.figure(figsize=(30, 10))
    ddata, dist_ = fancy_dendrogram(
                                    Z,
                                    #leaf_rotation=90.,
                                    #leaf_font_size=12.,
                                    show_contracted=True,
                                    annotate_above=2.,  # useful in small plots so annotations don't overlap
                                    )
    plt.xticks([])
    plt.savefig(savename_hc + 'tree_plot.png')

    # output the merger node list
    dist_ = np.sort(np.array(dist_))
    f = open(savename_hc + 'merge_node.txt', 'w')
    for i in dist_:
        print(i, file=f)
    f.close()
#timer
print('\n', '## CODE RUNTIME:', time.time()-Tstart) #Timer end
