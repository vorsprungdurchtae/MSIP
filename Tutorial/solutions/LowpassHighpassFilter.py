#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import img_as_float


def lowPassFilter(inputImage, r):
    m, n = inputImage.shape
    # Apply FFT and shift so that the lowest frequency is at the central pixel.
    Ff = np.fft.fftshift(np.fft.fft2(inputImage))
    # Mask the frequencies of the FFT based on their distance to the central pixel.
    x, y = np.meshgrid(range(n), range(m))
    frequencies = np.square(x-m/2) + np.square(y-n/2)
    Ff2 = np.multiply(frequencies < r**2, Ff)
    # Shift back, transform back and discard the imaginary component.
    output_image = np.real(np.fft.ifft2(np.fft.ifftshift(Ff2)))
    return output_image


def highPassFilter(inputImage, r1, r2):
    m, n = inputImage.shape
    # Apply FFT and shift so that the lowest frequency is at the central pixel.
    Ff = np.fft.fftshift(np.fft.fft2(inputImage))
    # Mask the frequencies of the FFT based on their distance to the central pixel.
    x, y = np.meshgrid(range(n), range(m))
    frequencies = np.square(x-m/2) + np.square(y-n/2)
    Ff2 = np.multiply(np.logical_and(
        frequencies < r2**2, frequencies > r1**2), Ff)
    # Shift back, transform back and discard the imaginary component.
    output_image = np.real(np.fft.ifft2(np.fft.ifftshift(Ff2)))
    return output_image


def main():
    # Read data
    input_image = imread('../figures/peppers.png', as_gray=True)
    input_image = img_as_float(input_image)

    # Frequency limits for the filters
    r = input_image.shape[0]//6
    print(r)
    r2 = 2*r

    # Create and apply different filters to the image
    high_pass_filtered_image = highPassFilter(input_image, r, r2)
    low_pass_filtered_image = lowPassFilter(input_image, r)

    # Plot the input image
    plt.subplot(1, 3, 1)
    plt.imshow(input_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Input image')

    plt.subplot(1, 3, 2)
    plt.imshow(low_pass_filtered_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Low-pass')

    plt.subplot(1, 3, 3)
    plt.imshow(high_pass_filtered_image, interpolation='nearest',
               cmap=plt.cm.get_cmap('gray'))
    plt.axis('off')
    plt.title('High-pass')
    plt.show()


if __name__ == '__main__':
    main()
