#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os

filedir = os.path.dirname(os.path.abspath(__file__))


def bin_histogram(image, num_bins):
    min_val = image.min()
    max_val = image.max()

    # For each pixel, compute the index of the bin the corresponding gray belongs to.
    histo_ind = (np.floor(
        (image.ravel() - min_val) * (num_bins - 1) / (max_val - min_val))).astype(int)

    # Straight forward implementation to compute the histogram from the indices with a for loop.
    # num_pix = image.size
    # histo = np.zeros(num_bins, dtype=int)
    # for i in range(num_pix):
    #     histo[histo_ind[i]] = histo[histo_ind[i]] + 1

    # Fast implemention avoiding the for loop using numpy
    histo = np.bincount(histo_ind)
    return histo


def equalize_histo(image, num_bins):
    m, n = image.shape
    min_val = image.min()
    max_val = image.max()
    num_pixels = m*n
    H_f = bin_histogram(image, num_bins)
    G_f = np.cumsum(H_f)

    histo_ind = (np.floor((image[:, :] - min_val) *
                          (num_bins - 1) / (max_val - min_val))).astype(int)
    image = (max_val - min_val) * G_f[histo_ind[:, :]] / num_pixels + min_val
    return image


def main():
    # Read data
    input_image = imread(filedir+'/cameraman.jpg', as_gray=True)/255

    equalized_image = equalize_histo(input_image, 255)

    # Plot the input image
    plt.subplot(2, 2, 1)
    plt.imshow(input_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.title('Input image')
    plt.axis('off')

    # Plot the histogram of the input image
    H_f = bin_histogram(input_image, 256)
    plt.subplot(2, 2, 2)
    plt.bar(range(H_f.size), H_f, color='C0')
    ax1 = plt.gca()
    ax1.tick_params('y', colors='C0')
    ax2 = plt.twinx()
    ax2.plot(np.cumsum(H_f), color='C1')
    ax2.tick_params('y', colors='C1')
    ax2.set_ylim([0, None])
    plt.title('$H_f$ and $G_f$ of input')

    # Plot the equalized image
    plt.subplot(2, 2, 3)
    plt.imshow(equalized_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.title('equalized image')
    plt.axis('off')

    # Plot the histogram of the equalized image
    H_f = bin_histogram(equalized_image, 256)
    plt.subplot(2, 2, 4)
    plt.bar(range(H_f.size), H_f, color='C0')
    ax1 = plt.gca()
    ax1.tick_params('y', colors='C0')
    ax2 = plt.twinx()
    ax2.plot(np.cumsum(H_f), color='C1')
    ax2.tick_params('y', colors='C1')
    ax2.set_ylim([0, None])
    plt.title('$H_f$ and $G_f$ of equalized image')
    plt.show()


if __name__ == '__main__':
    main()
