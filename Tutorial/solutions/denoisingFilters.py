#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float
from scipy.signal import correlate2d
from scipy.special import comb
from skimage.util import random_noise
from numba import jit
import os

filedir = os.path.dirname(os.path.abspath(__file__))


def filter_image(image, kernel, direct=True):
    if direct:
        m, n = image.shape
        c, d = kernel.shape
        a = int((c - 1) / 2)
        b = int((d - 1) / 2)
        padded_image = np.pad(image, (a, b))
        output_image = np.zeros((m, n))
        # Straightforward loops to compute the correlation
        for i in range(m):
            for j in range(n):
                sub_image = padded_image[i:i+c, j:j+d]
                output_image[i, j] = np.dot(sub_image.ravel(), kernel.ravel())
    else:
        # Python way of computing the correlation
        output_image = correlate2d(image, kernel, mode='same')
    return output_image


def mean_filter_kernel(a):
    kernel = np.ones((2*a+1, 2*a+1))
    kernel = kernel / np.sum(kernel)
    return kernel


def gauss_filter_kernel(a, sigma):

    # Straightforward loops to compute the kernel
    # kernel = np.zeros((2*a+1, 2*a+1))
    # for i in range(-a, a+1):
    #     for j in range(-a, a+1):
    #         kernel[i+a, j+a] = np.exp(- (i**2 + j**2) / (2 * sigma**2))

    # Python way of computing the kernel
    indX, indY = np.meshgrid(range(-a, a+1), range(-a, a+1))
    kernel = np.exp(- (np.square(indX) + np.square(indY)) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def binomial_filter_kernel(a):
    vec = np.zeros((2*a+1,))
    for i in range(2*a+1):
        vec[i] = comb(2*a, i)
    kernel = np.tensordot(vec, vec, axes=0)
    kernel = kernel / np.sum(kernel)
    return kernel


@jit(nopython=True)
def median_filter_image(image, a):
    m, n = image.shape
    output_image = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            start_X = max(0, i - a)
            stop_X = min(m, i + a + 1)
            start_Y = max(1, j - a)
            stop_Y = min(n, j + a + 1)
            sub_image = image[start_X:stop_X, start_Y:stop_Y]
            output_image[i, j] = np.median(sub_image)

    return output_image


def main():
    # Read data
    true_input_image = imread(filedir+'/../figures/peppers.png', as_gray=True)
    # Convert to float and value range [0,1]
    true_input_image = img_as_float(true_input_image)
    input_image = random_noise(true_input_image, mode="poisson")
    # input_image = random_noise(true_input_image, mode="gaussian")
    # input_image = random_noise(true_input_image, mode="s&p")

    # Filter radius
    a = 2

    # Create and apply different filters to the image
    mean_filtered_image = filter_image(input_image, mean_filter_kernel(a))
    gauss_filtered_image = filter_image(input_image, gauss_filter_kernel(a, 1))
    binom_filtered_image = filter_image(input_image, binomial_filter_kernel(a))

    # Duto is a convex combination of the original image and a smoothed version of it
    duto_filtered_image = 0.25 * input_image + 0.75 * gauss_filtered_image

    # Apply median filter to the image
    median_filtered_image = median_filter_image(input_image, a)

    # Plot the input image
    plt.subplot(2, 3, 1)
    plt.imshow(input_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Input image')

    plt.subplot(2, 3, 2)
    plt.imshow(mean_filtered_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Mean filtered image')

    plt.subplot(2, 3, 3)
    plt.imshow(gauss_filtered_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Gauss filtered image')

    plt.subplot(2, 3, 4)
    plt.imshow(binom_filtered_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Binomial filtered image')

    plt.subplot(2, 3, 5)
    plt.imshow(duto_filtered_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Duto filtered image')

    plt.subplot(2, 3, 6)
    plt.imshow(median_filtered_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Median filtered image')

    plt.show()


if __name__ == '__main__':
    main()
