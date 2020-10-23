# Unsupervised-ML-on-Galaxy-Morphology-using-VQ-VAE
This code is used in the paper: [Beyond the Hubble Sequence -- Exploring Galaxy Morphology with Unsupervised Machine Learning](https://arxiv.org/abs/2009.11932).  

We apply an unsupervised machine learning techniques consisting of **Vector-Quantised Variational Autoencoder (VQ-VAE)**[[1]](https://arxiv.org/abs/1711.00937)[[2]](https://arxiv.org/pdf/1906.00446.pdf) for feature learning and **Hierarchical Clustering (HC)** to explore galaxy morphology. In this work, we modify the original architecture of the VQ-VAE by adding an extra loss, which is defined by the silhouette score measured from a preliminary clustering result using K-medoids clustering (details please check our paper).

In this repo, we showcase the modified VQ-VAE code for feature learning used in this work and a simplified version of HC code to give an idea of how the code works. Here we attach 1000 SDSS DR7 images (data.hdf5), which we have done feature selection (1 sigma clip) and rescaling, to test the VQ-VAE code (vqvae.py). To save processing time, extracting features from images using a pre-trained VQ-VAE model (one provided in *pretrain_vqvae_model/*) and save them into a FITS file (vqvae_extract_feature.py). The HC code (hc.py) can then to be trained using the extracted features from images (FITS file).

## Easy guideline
- Extract features using *vqvae.py*
- Save the extracted features into a FITS file by *vqvae_extract_feature.py*
- Plot the HC merger tree using *hc.py* (Train=False)
- Train the HC using *hc.py* (Train=True)
- Check the output examples of each group :-D

# Software requirement
In addition to the basic libraries such as numpy, scipy, os, matplotlib, we need:
- tensorflow==1.14.0, tensorflow_probability==0.7.0
- [dm-sonnet==1.34](https://github.com/deepmind/sonnet)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [scikit-learn-extra](https://scikit-learn-extra.readthedocs.io/en/latest/install.html)
- [h5py](https://docs.h5py.org/en/latest/build.html)

# Citation
If using the concept proposed in the paper, or codes based on our code here, [please cite our paper](https://ui.adsabs.harvard.edu/abs/2020arXiv200911932C/abstract).
