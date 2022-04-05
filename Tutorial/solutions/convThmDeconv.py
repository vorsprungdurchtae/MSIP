#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from skimage.io import imread
from skimage import img_as_float
import matplotlib.pyplot as plt
from denoisingFilters import filter_image, gauss_filter_kernel


def Deconvolve(input_image, kernel, eps):
    m, n = input_image.shape

    # The central pixel of our kernel corresponds to the origin.
    # We have to zero-pad and shift the kernel accordingly.
    c, d = kernel.shape
    a = (c - 1) // 2
    b = (d - 1) // 2
    padded_kernel = np.zeros((m, n))
    padded_kernel[(m//2-a):(m//2+a)+1, (n//2-b):(n//2+b)+1] = kernel
    padded_kernel = np.fft.ifftshift(padded_kernel)
    Ff = np.fft.fft2(input_image)
    Fpsi = np.fft.fft2(padded_kernel)

    div = np.divide(np.multiply(Ff, np.conj(Fpsi)), np.multiply(np.conj(Fpsi), Fpsi) + eps)

    output_image = np.real(np.fft.ifft2(div))
    return output_image


if __name__ == '__main__':
    # Read data
    inputImage = imread('../figures/peppers.png', as_gray=True)
    inputImage = img_as_float(inputImage)

    # Filter radius and strength
    a = 20
    sigma = 1.5

    # Create kernel, filter the image.
    kernel = gauss_filter_kernel(a, sigma)
    filtered_image = filter_image(inputImage, kernel)
    quantized_filtered_image = ((255*filtered_image).astype(np.uint8))/255

    # Regularization parameter of the complex division
    epsilon = 1e-4

    deconvolved_image = Deconvolve(filtered_image, kernel, epsilon)
    deconvolved_quantized_image = Deconvolve(quantized_filtered_image, kernel, epsilon)

    # Plot the input image
    plt.subplot(2, 2, 1)
    plt.imshow(inputImage, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Input image')

    plt.subplot(2, 2, 2)
    plt.imshow(filtered_image, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Filtered image')

    plt.subplot(2, 2, 3)
    plt.imshow(deconvolved_image, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Deconvolved image')

    plt.subplot(2, 2, 4)
    plt.imshow(deconvolved_quantized_image, interpolation='nearest', cmap=plt.cm.get_cmap('gray'), vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Deconvolved quantized image')
    plt.show()
