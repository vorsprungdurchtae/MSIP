#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from skimage.io import imread, imsave
from skimage import img_as_float
import numpy as np
from pathlib import Path
from scipy.signal import correlate2d
from skimage.color import hsv2rgb
import os

filepath = os.path.dirname(os.path.abspath(__file__))


def PrewittSobelFilterKernel(direction, mode='Prewitt'):
    diff = np.array([-1, 0, 1]) / 2
    if mode == 'Prewitt':
        mean = np.array([1, 1, 1]) / 3
    elif mode == 'Sobel':
        mean = np.array([1, 2, 1]) / 4
    else:
        raise ValueError('Invalid mode')

    if direction == 1:
        return np.tensordot(diff, mean, axes=0)
    else:
        return np.tensordot(mean, diff, axes=0)


def main():
    inputFilename = filepath+'/../figures/cvtest.png'
    inputImage = imread(inputFilename, as_gray=True)
    inputImage = img_as_float(inputImage)

    # Create and apply different filters to the image
    DX = correlate2d(inputImage, PrewittSobelFilterKernel(
        0, 'Sobel'), mode='same', boundary='symm')
    DY = correlate2d(inputImage, PrewittSobelFilterKernel(
        1, 'Sobel'), mode='same', boundary='symm')

    # Compute gradient size and directions (as angles)
    size = np.hypot(DX, DY)
    size /= np.amax(size)
    angles = -np.arctan2(DY, DX)

    outputFilename = Path(inputFilename).stem
    imsave(outputFilename + '_DX.png', DX)
    imsave(outputFilename + '_DY.png', DY)
    imsave(outputFilename + '_gradSize.png', size)

    colorImage = np.zeros(inputImage.shape+(3,))
    colorImage[..., 0] = np.remainder(angles + 2*np.pi, 2*np.pi) / (2*np.pi)
    colorImage[..., 1] = 1
    colorImage[..., 2] = 1
    imsave(outputFilename + '_gradDirGray.png', colorImage[..., 0])
    imsave(outputFilename + '_gradDirH11.png', hsv2rgb(colorImage))
    colorImage[..., 1] = size
    imsave(outputFilename + '_gradDirHS1.png', hsv2rgb(colorImage))
    colorImage[..., 2] = size
    imsave(outputFilename + '_gradDirHSV.png', hsv2rgb(colorImage))


if __name__ == '__main__':
    main()
