import os
import sys
import glob
import h5py
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import imageio
import skopi as sk
import pickle

import time
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from scipy.spatial.transform import Rotation as R

from utils import *


class CDI_Experiment :
    
    def __init__(self, data_path, plot_dir, geom_path):
        """
        Initializes an instance of the CDI_Experiment class
        
        :param data_path: path to .h5 data file
        :param plot_dir: directory where future plots will be saved
        :param geom_path: path to detector file
        """
        
        self.data_path = data_path
        self.data = h5py.File(self.data_path, 'r')
        self.det = sk.PnccdDetector(geom=geom_path)
        
        self.n_images, self.n_panels, self.d_panel, _ = self.data['intensities'].shape
        self.d, self.d_slit = self.det.assemble_image_stack(self.data['intensities'][0]).shape
        
        self.plot_dir = plot_dir
        check_create(self.plot_dir)
    
    
    # Data processing functions
    
    
    def init_iPCA(n_batches, n_comp, d_crop, sub_ratio) :
        """
        Initializes an incremental PCA
        
        :param n_batches: number of batches to divide data into
        :param n_comp: number of components to keep in the iPCA
        :param d_crop: dimension to crop images into
        :param sub_ratio: sub_sampling ratio of data after cropping. A ratio of 1 means that no subsampling is applied
        """
        
        self.n_batches = n_batches
        self.batch_size = self.n_images // self.n_batches
        self.batch_index = np.arange(self.n_images).reshape(self.n_batches, self.batch_size)

        self.n_comp = n_comp

        self.d_crop = d_crop
        self.sub_ratio = sub_ratio
        self.sub_d = self.d_crop // self.sub_ratio
        self.sub_index = self.sub_ratio * np.arange(self.sub_d)
        sub_row = np.full(self.d_crop, False)
        sub_row[self.sub_index] = True

        crop_mask = np.full((self.d_crop, self.d_crop), False)
        crop_mask[self.sub_index, :] = sub_row
        det_mask = np.full((self.d, self.d), False)
        det_mask[(self.d - self.d_crop) // 2:(self.d + self.d_crop) // 2, (self.d - self.d_crop) // 2:(self.d + self.d_crop) // 2] = crop_mask
        slit_mask = np.full((self.d, self.d_slit - self.d), False)
        self.det_mask = np.concatenate((det_mask[:, :self.d_panel], slit_mask, det_mask[:, self.d_panel:]), axis=1)

        self.pca = IncrementalPCA(n_components=self.n_comp)
        self.coordinates = []
        
    
    def train_iPCA() :
        """
        Trains the iPCA batch by batch
        """
        
        for index in tqdm(self.batch_index) :
            batch = self.det.assemble_image_stack_batch(self.data['intensities'][index])[:, self.det_mask]
            self.pca.partial_fit(batch)
        
        self.eigenimages = self.pca.components_.reshape(self.n_comp, self.sub_d, self.sub_d)
        return
    
    
    def transform_data() :
        """
        Applies PCA to the training dataset
        """
        
        for index in tqdm(batch_index) :
            batch_coord = self.pca.transform(self.det.assemble_image_stack_batch(self.data['intensities'][index])[ :, self.det_mask])
            self.coordinates.append(batch_coord)

        self.coordinates = np.concatenate(self.coordinates)
        return
        
    
    def save_iPCA(pca_path, coord_path) :
        """
        Saves the iPCA and the transformed dataset by pickling them
        
        :param pca_path: saving location for the pca
        :param coord_path: saving location for the transformed dataset
        """
        
        pca_file = open(pca_path,'wb')
        pickle.dump(self.pca, pca_file)
        pca_file.close()
        
        coord_file = open(coord_path,'wb')
        pickle.dump(self.coordinates, coord_file)
        coord_file.close()
        return
        
    
    def load_iPCA(pca_path, coord_path) :
        """
        Loads a pickled iPCA file and its transformed training dataset
        
        :param pca_path: path to the pca file
        :param coord_path: path to transformed data
        """
        
        pca_file = open(pca_path,'rb')
        self.pca = pickle.load(pca_file)
        pca_file.close()
        
        coord_file = open(coord_path,'rb')
        self.coordinates = pickle.load(coord_file)
        coord_file.close()
        return
    
    def reconstruct_data(n_partial=3) :
        """
        Reconstructs the initial dataset from its transformed version by inverse PCA-transformation. Two reconstructions
        are built : either using all components of the iPCA, or only the first n_partial ones
        
        :param n_partial: number of components of the iPCA to consider when computing the partial reconstruction
        """

        self.full_reconstruct = self.pca.inverse_transform(self.coordinates).reshape(self.n_images, self.sub_d, self.sub_d)
        self.partial_reconstruct = (np.dot(self.coordinates[:, :n_partial], self.pca.components_[:n_partial]) + self.pca.mean_).reshape(self.n_images, self.sub_d, self.sub_d)
        return

    
    def ica(n_comp=2, random_state=92) :
        """
        Applies ICA to the PCA-transformed dataset in order to straighten the shape of its first three components.
        This makes binning easier.
        
        :param n_comp: number of components of the ICA
        """
        
        ica_ = FastICA(n_components=n_comp, random_state=random_state)
        self.ica_coordinates = ica_.fit_transform(self.coordinates[:, 1:3])
        return
    
    
    def binning(n_bins) :
        """
        Binning of the shape created by the first three components of the ICA-PCA-transformed dataset. We bin by increasing
        order of the third component of the ICA. We also compute in each bin the angle between the rotation axes associated
        with each of its points and its center axis as defined in notebook cdi_process.ipynb.
        
        :param n_bins: number of bins to divide the dataset into
        """
        
        # Initializes binning
        self.n_bins = n_bins
        min_bin = self.ica_coordinates[:, 1].min()
        max_bin = self.ica_coordinates[:, 1].max()
        bins = np.linspace(min_bin, max_bin, n_bins + 1)
        self.bin_indexes = np.array([(self.ica_coordinates[:, 1] >= self.bins[i]) * (self.ica_coordinates[:, 1] < self.bins[i + 1]) for i in range(self.n_bins)])
        
        # Computes correlations and angles among each pair of point in each bin
        self.cors = [bin_cor(b) for b in self.bin_indexes]
        center_axes = [find_center_axis(cor) for cor in cors]
        self.angles = []
        for i in range(n_bins) :
            b = self.bin_indexes[i]
            len_bin = b.sum()
            axis = center_axes[i]
            quats = self.data['orientations'][:][b]
            quat1 = quats[axis]
            theta = np.zeros(len_bin)
            for j in range(len_bin) :
                quat2 = quats[j]
                theta[j] = diff_angle(quat1, quat2)
            self.angles.append(theta)
        
        # Determines the argument of the complex number formed by the the first two PCs of points in each bin and sorts
        # points according to this measure
        bin_angles = [angle_array(coordinates[b, :2]) for b in bin_indexes]
        self.bin_angle_indexes = [np.argsort(bin_angles[i]) for i in range(n_bins)]
        return
    
    
    # Plot functions
    
    
    def EVR_plot() :  
        """
        Plots the explained variance ratio of the components of the iPCA
        """
        
        fig = plt.figure(figsize=(6, 4))
        plt.bar(np.arange(n_comp), self.pca.explained_variance_ratio_)
        plt.title('Explained Variance Ratio')
        plt.show()
        return fig
    
    
    def PC_plot(coord, pc_i=0, pc_j=1, is_ICA=False) :
        """
        Scatter plot of the transformed dataset's pc_i column against its pc_j column

        :param coord: any two-dimensional array
        :param pc_i: first column to plot
        :param pc_j: second column to plot
        :param is_ICA: if True, we consider the dataset after ICA-transformation
        """
        
        if is_ICA :
            coord = self.ica_coordinates
        else :
            coord = self.coordinates

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(coord[:, pc_i], coord[:, pc_j])
        plt.xlabel(f'PC{pc_i + 1}')
        plt.ylabel(f'PC{pc_j + 1}')
        plt.close()
        return fig
    
    
    def PC_plot3d(pc_i=0, pc_j=1, pc_k=2, angle=0, is_ICA=False) :
        """
        3D scatter plot of the transformed dataset's pc_i column against its pc_j and pc_k columns

        :param pc_i: first column to plot
        :param pc_j: second column to plot
        :param pc_k: third column to plot
        :param angle: viewing angle for the 3d plot
        :param is_ICA: if True, we consider the dataset after ICA-transformation
        """
        
        if is_ICA :
            coord = self.ica_coordinates
        else :
            coord = self.coordinates

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(coord[:, pc_i], coord[:, pc_j], coord[:, pc_k], alpha=0.2)
        ax.set_xlabel(f'PC{pc_i + 1}')
        ax.set_ylabel(f'PC{pc_j + 1}')
        ax.set_zlabel(f'PC{pc_k + 1}')
        ax.view_init(30, angle)
        plt.close()
        return fig
    
    
    def eigenimages_plot(nrows=10, ncols=5) :
        """
        Plots the eigenimages of the iPCA
        """

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15*(nrows//ncols)))
        for i in range(nrows):
            for j in range(ncols):
                image = self.eigenimages[i*ncols+j]
                axs[i,j].imshow(np.abs(image), norm=LogNorm(), interpolation='none', cmap='Greys_r')
        plt.close()
        return fig
    
    
    def compare_plot(n_samples=10, random_state=92) :
        """
        Compares n_samples images from the original training dataset with their full and partial reconstructions
        """
        
        np.random.seed(random_state)
        samples = np.random.choice(np.arange(self.n_images), size=n_samples, replace=False)
        
        nrows, ncols = n_samples, 3
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15*(nrows//ncols)))
        for i in range(n_samples):
            idx = samples[i]
            image = self.det.assemble_image_stack(self.data['intensities'][idx])[self.det_mask].reshape(self.sub_d, self.sub_d)
            fullr = self.full_reconstruct[idx]
            partialr = self.partial_reconstruct[idx]
            axs[i,0].imshow(image, norm=LogNorm(), interpolation='none')
            axs[i,1].imshow(np.abs(fullr), norm=LogNorm(), interpolation='none', cmap='Greys_r')
            axs[i,2].imshow(np.abs(partialr), norm=LogNorm(), interpolation='none', cmap='Greys_r')
        plt.close()
        return fig
    
    
    def bin_plot(n_cols=3) :
        """
        Scatter plot for each bin of the first two PCs of its points. Points are coloured according to the angle
        between them and the center axis of their bin.
        """
        
        nrows = int(np.ceil(n_bins/ncols))

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15*(nrows//ncols)))
        for i in range(nrows):
            for j in range(ncols) :
                if i * ncols + j < n_bins :
                    b = self.bin_indexes[i * ncols + j]
                    im = axs[i, j].scatter(self.coordinates[b][:, 0], self.coordinates[b][:, 1], c=self.angles[i * ncols + j], cmap='twilight_shifted')

        fig.colorbar(im, ax=axs.ravel().tolist())
        plt.close()
        return fig
    
    
    def BinImagePlot(b, k_bin) :
        """
        Side-by-side plot of :
        - a scatter plot of the first two PCs of bin b's points. Point k_bin is highlighted in red
        - the original image corresponding to point k_bin
        
        :param b: numpy array which indexes points belonging to a certain bin
        :param k_bin: index of the point to consider. Points of a bin are indexed in increasing order of argument (as
        computed when binning)
        """
        true_index = np.arange(n_images)[b][k_bin]
        image = self.det.assemble_image_stack(self.data['intensities'][true_index])[self.det_mask].reshape(self.sub_d, self.sub_d)
        k_mask = (np.arange(b.sum()) == k_bin)
        not_k_mask = np.logical_not(k_mask)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax1.scatter(self.coordinates[b][not_k_mask, 0], self.coordinates[b][not_k_mask, 1], color='blue', alpha=0.3)
        ax1.scatter(self.coordinates[b][k_mask, 0], self.coordinates[b][k_mask, 1], s=200, color='r', marker='^')
        ax2.imshow(image, norm=LogNorm(), interpolation='none')

        ax1.set_title('Bin Plot')
        ax1.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('Original Image')
        plt.close()
        return fig
    
    
    def bin_lap_gif(b_idx, n_frames) :
        """
        Creates and saves a GIF animation of frames created using BinImagePlot when going through a bin in increasing
        order of argument (as computed when binning)
        
        :param b_idx: index of bin to consider
        :param n_frames: number of frames of the gif
        """
        
        self.gif_dir = os.path.join(self.plot_dir, 'gif/')
        check_create(self.gif_dir)
        
        self.bin_lap_dir = os.path.join(self.gif_dir, 'bin_lap/')
        check_create(self.bin_lap_dir)

        b = self.bin_indexes[b_idx]
        len_bin = b.sum()
        frames = np.linspace(0, len_bin, num=n_frames, endpoint=False, dtype='int32')
        frames = self.bin_angle_indexes[b_idx][frames]

        gif_name = os.path.join(self.gif_dir, f'bin{b_idx}_subratio{self.sub_ratio}.gif')
        images = []

        for k in tqdm(frames) :
            filename = os.path.join(self.bin_lap_dir, f'{k}.png')
            BinImagePlot(b, k).savefig(filename)
            images.append(imageio.imread(filename))
            os.remove(filename)

        imageio.mimsave(gif_name, images, fps=10)
        return