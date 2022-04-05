#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from skimage.io import imread
from skimage import img_as_float
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import os

filepath = os.path.dirname(os.path.abspath(__file__))


def CannyEdges(image, sigma, theta, a, b):
    m, n = image.shape
    # Compute the Gauss kernel
    indX, indY = np.meshgrid(range(-b, b+1), range(-a, a+1))
    G = np.exp(- (np.square(indX) + np.square(indY)) / (2 * sigma**2))
    G = G / np.sum(G)
    # Compute the Gauss kernel derivatives
    G_x = np.multiply(G, -indX / (sigma**2))
    G_x = G_x - np.mean(G_x)
    G_y = np.multiply(G, -indY / (sigma**2))
    G_y = G_y - np.mean(G_y)
    # Compute x and y derivative of the image by filtering
    I_x = correlate2d(image, -G_x, mode='same')
    I_y = correlate2d(image, -G_y, mode='same')
    # Compute the norm of the image gradient at each pixel,
    # padded with "-1", so that we have neighboring rho values
    # in all directions at each image pixel.
    rho = np.pad(np.hypot(I_x, I_y), (1, 1),
                 mode='constant', constant_values=-1)
    # Compute angles corresponding to the gradient direction.
    Theta = np.arctan2(I_y, I_x)
    # Round angles to multiples of 45 degrees
    Theta = np.round(8*(Theta+np.pi)/(2*np.pi)) * 2*np.pi / 8 - np.pi
    # Compute directions for each pixel for the Canny condition
    dir = np.zeros((m, n, 2))
    dir[:, :, 0] = np.sin(Theta)
    dir[:, :, 1] = np.cos(Theta)
    dir = np.round(dir).astype(int)
    # Check the Canny condition at each pixel
    outputImage = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if ((rho[i+1, j+1] >= theta) and
                (rho[i+1, j+1] > max(rho[i+1+dir[i, j, 0], j+1+dir[i, j, 1]],
                                     rho[i+1-dir[i, j, 0], j+1-dir[i, j, 1]]))):
                outputImage[i, j] = 1

    return outputImage


# Read data
inputImage = imread(filepath+'/../figures/peppers.png', as_gray=True)
inputImage = img_as_float(inputImage)

# Set parameters
a = 9
b = 9
sigma = 0.75
theta = 0.05

# Compute the Canny edges
C = CannyEdges(inputImage, sigma, theta, a, b)

plt.subplot(1, 2, 1)
plt.imshow(inputImage, interpolation='nearest',
           cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
plt.axis('off')
plt.title('Input image')

plt.subplot(1, 2, 2)
plt.imshow(C, interpolation='nearest',
           cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
plt.axis('off')
plt.title('Canny edges')
plt.show()
