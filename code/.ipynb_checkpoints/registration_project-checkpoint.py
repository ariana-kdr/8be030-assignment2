"""
Project code for image registration topics.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import registration as reg
from IPython.display import display, clear_output


def intensity_based_registration_demo():

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread('../data/image_data/1_2_t1.tif')
    Im = plt.imread('../data/image_data/1_2_t1_d.tif')        

    # initial values for the parameters
    # we start with the identity transformation
    # most likely you will not have to change these
    x = np.array([0., 0., 0.])          # Rigid
    # x = np.array([0., 1., 1., 0., 0., 0., 0. ])

    
    # NOTE: for affine registration you have to initialize
    # more parameters and the scaling parameters should be
    # initialized to 1 instead of 0

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation
    fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=False)
    # fun = lambda x: reg.affine_corr(I, Im, x, return_transform=False)
    # fun = lambda x: reg.affine_mi(I, Im, x, return_transform=False)

    # the learning rate
    mu = 0.001
    
    # number of iterations
    num_iter = 200

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.3, 0.95,
        np.array2string(x, precision=5, floatmode='fixed'),
        bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10},
        transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = reg.rigid_corr(I, Im, x, return_transform=True)
        # S, Im_t, _ = reg.affine_corr(I, Im, x, return_transform=True)
        # S, Im_t, _ = reg.affine_mi(I, Im, x, return_transform=True)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)



def get_params(I, Im, similarity):
    # Helper function to set up inputs for intensity_based_registration()
    # INPUT:
    #     I          - the fixed image 
    #     Im         - the moving image
    #     similarity - which measure is being used, EITHER rigid_corr OR affine_corr OR affine_mi
    # OUTPUT:
    #     x    - the correct array, either for rigid or affine 
    #     name - the name of the similarity measure being used as a string, to put in the plots later
    #     fun  - the call for the similarity measure function
    
    if similarity == 'rigid_corr':
        x = np.array([0., 0., 0.])
        name = "Rigid NCC" 
        fun = lambda x: reg.rigid_corr(I, Im, x, return_transform=True)

    elif similarity == 'affine_corr':
        x = np.array([0., 1., 1., 0., 0., 0., 0. ])
        name = "Affine NCC"
        fun = lambda x: reg.affine_corr(I, Im, x, return_transform=True)

    elif similarity == 'affine_mi':
        x = np.array([0., 1., 1., 0., 0., 0., 0. ])
        name = "Affine MI" 
        fun = lambda x: reg.affine_mi(I, Im, x, return_transform=True)

    else:
        print("Error, incorrect similarity metric")
            
    return x, name, fun


def intensity_based_registration(I, Im, x, mu, num_iter, name, fun):
    # Same function as above, but modularised to take arguments 
    # INPUT: 
    #     I        - the fixed image 
    #     Im       - the moving image
    #     x        - EITHER x_rigid = np.array([0., 0., 0.]) OR x_affine = np.array([0., 1., 1., 0., 0., 0., 0. ])
    #     mu       - learning rate 
    #     num_iter - number of iterations
    #     name     - name of similarity measure being used, EITHER "Rigid NCC" OR "Affine NCC" OR "Affine MI" 
    #     fun      - EITHER reg.rigid_corr OR reg.affine_corr OR reg.affine_mi
    # OUTPUT:
    #     no return, but displays two updating plots (the overlaid images and similarity plot)
    
    fun_g = lambda x: fun(x)[0]
    
    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    fig = plt.figure(figsize=(14,6))

    # fixed and moving image, and parameters
    ax1 = fig.add_subplot(121)

    # fixed image
    im1 = ax1.imshow(I)
    # moving image
    im2 = ax1.imshow(I, alpha=0.7)
    # parameters
    txt = ax1.text(0.05, 0.95, np.array2string(x, precision=5, floatmode='fixed'),
                   fontsize=7,
                   horizontalalignment='left',
                   verticalalignment='bottom',                   
                   bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
                   transform=ax1.transAxes)

    # 'learning' curve
    ax2 = fig.add_subplot(122, xlim=(0, num_iter), ylim=(0, 1))

    learning_curve, = ax2.plot(iterations, similarity, lw=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Similarity')
    ax2.grid()
    # Add fixed parameters as text box
    param_text = (f'μ = {mu}\n'
                  f'iter = {num_iter}\n'
                  f'{name}')
    ax2.text(0.05, 0.95, param_text,
             fontsize=9,
             horizontalalignment='left',
             verticalalignment='top',
             bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
             transform=ax2.transAxes)

    # perform 'num_iter' gradient ascent updates
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun_g, x)
        x += g*mu

        # for visualization of the result
        S, Im_t, _ = fun(x)

        clear_output(wait = True)

        # update moving image and parameters
        im2.set_data(Im_t)
        txt.set_text(np.array2string(x, precision=5, floatmode='fixed'))

        # update 'learning' curve
        similarity[k] = S
        learning_curve.set_ydata(similarity)

        display(fig)
        
    plt.close(fig)  # prevents Jupyter from showing the figure a second time


def gaussian_noise(img, mean, st_dev, display):
    # Add Gaussian noise to input image 
    # INPUT:
    #     img     - input image to add noise to 
    #     mean    - mean of Gaussian distribution 
    #     st_dev  - standard deviation of Gaussian distribution
    #     display - if 'yes', the original and noisy image get plotted
    # OUTPUT:
    #     img_noise - output image with Gaussian noise

    # Convert input image to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Draw random samples from a Gaussian distribution with the same size as the gray image 
    noise = np.random.normal(mean, st_dev, img_gray.shape)

    # Add noise to gray image 
    img_noise = img_gray + noise

    # Remove out of range colour values 
    img_clip = np.clip(img_noise, 0, 255)

    # Display both the input image and the image with Gaussian noise
    if display == 'yes':
        fig = plt.figure(figsize=(14,6))
    
        # Add input image on the left
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(img_gray)
        ax1.set_title('Original Image')
    
        # Add grey image on the right with mean and stdev info
        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(img_clip) #, cmap='gray')
        ax2.set_title(f'Gaussian Noise: μ={mean}, σ={st_dev}')

    return img_clip