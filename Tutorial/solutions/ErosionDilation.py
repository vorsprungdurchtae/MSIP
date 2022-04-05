#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from skimage.io import imread
from skimage import img_as_float
import numpy as np
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
from attendance2 import isodata_threshold
import os

filepath = os.path.dirname(os.path.abspath(__file__))

def erodeImage(image, a):
    # Straightforward loop solution
    # m, n = image.shape
    # outputImage = np.zeros((m, n))
    # for i in range(m):
    #     for j in range(n):
    #         startX = max(0, i - a)
    #         stopX = min(m, i + a + 1)
    #         startY = max(0, j - a)
    #         stopY = min(n, j + a + 1)
    #         subImage = image[startX:stopX, startY:stopY]
    #         outputImage[i, j] = np.amin(subImage)

    # More efficient solution using skimage.util.view_as_windows
    # Since we will compute the min on each patch, we can extend with the image's maximum.
    paddedImage = np.pad(image, (a, a), mode='constant', constant_values=np.amax(image))
    # view_as_windows is a convenient way to extract all overlapping patches from an image.
    view = view_as_windows(paddedImage, (2*a+1, 2*a+1), step=1)
    # now compute the patch wise minimum
    outputImage = np.amin(view, axis=(2, 3))

    return outputImage


def dilateImage(image, a):
    # Straightforward loop solution
    # m, n = image.shape
    # outputImage = np.zeros((m, n))
    # for i in range(m):
    #     for j in range(n):
    #         startX = max(0, i - a)
    #         stopX = min(m, i + a + 1)
    #         startY = max(0, j - a)
    #         stopY = min(n, j + a + 1)
    #         subImage = image[startX:stopX, startY:stopY]
    #         outputImage[i, j] = np.amax(subImage)

    # More efficient solution using skimage.util.view_as_windows
    # Since we will compute the max on each patch, we can extend with the image's minimum.
    paddedImage = np.pad(image, (a, a), mode='constant', constant_values=np.amin(image))
    view = view_as_windows(paddedImage, (2*a+1, 2*a+1), step=1)
    outputImage = np.amax(view, axis=(2, 3))

    return outputImage


def main():
    # Read data
    inputImage = imread(filepath+'/../figures/tafel_low.png', as_gray=True)
    inputImage = img_as_float(inputImage)

    # Filter radius
    a = 7

    # Create and apply different filters to the image
    erodedImage = erodeImage(inputImage, a)
    # The above is eqiuivalent to
    # erodedImage = scipy.ndimage.morphology.grey_erosion(inputImage, 2*a+1)
    dilatedImage = dilateImage(inputImage, a)
    openedImage = dilateImage(erodedImage, a)
    closedImage = erodeImage(dilatedImage, a)
    thresholdedImage, _ = isodata_threshold(inputImage - openedImage, 256)

    # Plot the input image
    plt.subplot(2, 3, 1)
    plt.imshow(inputImage, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Input image')

    plt.subplot(2, 3, 2)
    plt.imshow(erodedImage, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Eroded image')

    plt.subplot(2, 3, 3)
    plt.imshow(dilatedImage, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Dilated image')

    plt.subplot(2, 3, 4)
    plt.imshow(openedImage, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Opened image')

    plt.subplot(2, 3, 5)
    plt.imshow(closedImage, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Closed image')

    plt.subplot(2, 3, 6)
    plt.imshow(thresholdedImage, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Isodata result after background subtraction')
    plt.show()


if __name__ == '__main__':
    main()
