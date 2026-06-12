"""
Registration module main code.
"""

import numpy as np
from scipy import ndimage
import registration_util as util


# SECTION 1. Geometrical transformations


def identity():
    # 2D identity matrix.
    # Output:
    # T - transformation matrix

    T = np.eye(2)

    return T


def scale(sx, sy):
    # 2D scaling matrix.
    # Input:
    # sx, sy - scaling parameters
    # Output:
    # T - transformation matrix

    T = np.array([[sx,0],[0,sy]])

    return T


def rotate(phi):
    # 2D rotation matrix.
    # Input:
    # phi - rotation angle
    # Output:
    # T - transformation matrix

    #------------------------------------------------------------------#
    T = np.array([[np.cos(phi), -np.sin(phi)], 
                  [np.sin(phi),  np.cos(phi)]])
    #------------------------------------------------------------------#

    return T


def shear(cx, cy):
    # 2D shearing matrix.
    # Input:
    # cx - horizontal shear
    # cy - vertical shear
    # Output:
    # T - transformation matrix

    #------------------------------------------------------------------#
    T = np.array([[1.0, cx],
                  [ cy, 1.0]])
    #------------------------------------------------------------------#
    
    return T


def reflect(rx, ry):
    # 2D reflection matrix.
    # Input:
    # rx - horizontal reflection (must have value of -1 or 1)
    # ry - vertical reflection (must have value of -1 or 1)
    # Output:
    # T - transformation matrix

    allowed = [-1, 1]
    if rx not in allowed or ry not in allowed:
        T = 'Invalid input parameter'
        return T

    #------------------------------------------------------------------#
    T = np.array([[rx, 0],
                  [ 0, ry]])
    #------------------------------------------------------------------#
    
    return T


# SECTION 2. Image transformation and least squares fitting


def image_transform(I, Th,  output_shape=None):
    # Image transformation by inverse mapping.
    # Input:
    # I - image to be transformed
    # Th - homogeneous transformation matrix
    # output_shape - size of the output image (default is same size as input)
    # Output:
    # It - transformed image
	# Xt - remapped coordinates
    
    # we want double precision for the interpolation, but we want the
    # output to have the same data type as the input - so, we will
    # convert to double and remember the original input type

    input_type = type(I)

    # default output size is same as input
    if output_shape is None:
        output_shape = I.shape

    # spatial coordinates of the transformed image
    x = np.arange(0, output_shape[1])
    y = np.arange(0, output_shape[0])
    xx, yy = np.meshgrid(x, y)

    # convert to a 2-by-p matrix (p is the number of pixels)
    X = np.concatenate((xx.reshape((1, xx.size)), yy.reshape((1, yy.size))))
    # convert to homogeneous coordinates
    Xh = util.c2h(X)

    #------------------------------------------------------------------#
    # Perform inverse coordinates mapping.
    Th_inv = np.linalg.inv(Th)
    Xt = Th_inv.dot(Xh)         # x_transf = T^-1 . x_init
    #------------------------------------------------------------------#

    It = ndimage.map_coordinates(I, [Xt[1,:], Xt[0,:]], order=1, mode='constant').reshape(output_shape)

    return It, Xt


def ls_solve(A, b):
    # Least-squares solution to a linear system of equations.
    # Input:
    # A - matrix of known coefficients
    # b - vector of known constant term
    # Output:
    # w - least-squares solution to the system of equations
    # E - squared error for the optimal solution
    

    #------------------------------------------------------------------#
    # Implement the least-squares solution for w.
    w, ssr, rank, sv = np.linalg.lstsq(A, b, rcond=None)                      # since A is not a square matrix we can't use linalg.solve()

    # Compute the residuals, [r0, r1, r2, r3]
    res = A.dot(w) - b

    # Compute the individual errors (per solution), [r0², r1², r2², r3²]
    err = res**2
    #------------------------------------------------------------------#

    # compute the error for optimal solution, ||Aw - b||²
    # E = (Aw - b)ᵀ · (Aw - b) = ||Aw - b||² = r[0]² + r[1]² + r[2]² + r[3]²
    E = np.transpose(res).dot(res)                                             # adapted 'A.dot(w) - b' to 'res' 
    
    return w, E, err                                                           # adapted to also return individual errors 


def ls_affine(X, Xm):
    # Least-squares fitting of an affine transformation.
    # Input:
    # X - Points in the fixed image
    # Xm - Corresponding points in the moving image
    # Output:
    # T - affine transformation in homogeneous form.

    A = np.transpose(Xm)  # points in the moving image


    #------------------------------------------------------------------#
    # Implement least-squares fitting of an affine transformation.
    # Use the ls_solve() function that you have previously implemented.

    B = np.transpose(X[:2])   # points in the fixed image; B now has two columns [b1 b2] 

    w, _, _ = ls_solve(A, B)  # solve both systems 

    T = np.eye(3)
    T[:2] = w.T               # fill both rows of T with w  
    
    #------------------------------------------------------------------#

    return T


# SECTION 3. Image simmilarity metrics


def correlation(I, J):
    # Compute the normalized cross-correlation between two images.
    # Input:
    # I, J - input images
    # Output:
    # CC - normalized cross-correlation
    # it's always good to do some parameter checks

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    u = I.reshape((I.shape[0]*I.shape[1],1))
    v = J.reshape((J.shape[0]*J.shape[1],1))

    # subtract the mean
    u = u - u.mean(keepdims=True)
    v = v - v.mean(keepdims=True)

    #------------------------------------------------------------------#
    # Implement the computation of the normalized cross-correlation.
    # This can be done with a single line of code, but you can use for-loops instead.
    u_tr = np.transpose(u)
    v_tr = np.transpose(v)

    #CC = np.vdot(u_tr,v) / ( np.sqrt(np.vdot(u_tr,u)) * np.sqrt(np.vdot(v_tr,v)) )      # changed these first from np.dot to np.vdot

    CC = (u.T.dot(v))/(np.sqrt(u.T.dot(u))*np.sqrt(v.T.dot(v)))
    
    #------------------------------------------------------------------#

    return CC[0][0]


def joint_histogram(I, J, num_bins=16, minmax_range=None):
    # Compute the joint histogram of two signals.
    # Input:
    # I, J - input images
    # num_bins: number of bins of the joint histogram (default: 16)
    # range - range of the values of the signals (default: min and max of the inputs)
    # Output:
    # p - joint histogram

    if I.shape != J.shape:
        raise AssertionError("The inputs must be the same size.")

    # make sure the inputs are column-vectors of type double (highest precision)
    I = I.reshape((I.shape[0]*I.shape[1],1)).astype(float)
    J = J.reshape((J.shape[0]*J.shape[1],1)).astype(float)

    # if the range is not specified use the min and max values of the inputs
    if minmax_range is None:
        minmax_range = np.array([min(min(I),min(J)), max(max(I),max(J))])

    # this will normalize the inputs to the [0 1] range
    I = (I-minmax_range[0]) / (minmax_range[1]-minmax_range[0])
    J = (J-minmax_range[0]) / (minmax_range[1]-minmax_range[0])

    # and this will make them integers in the [0 (num_bins-1)] range
    I = np.round(I*(num_bins-1)).astype(int)
    J = np.round(J*(num_bins-1)).astype(int)

    n = I.shape[0]
    hist_size = np.array([num_bins,num_bins])

    # initialize the joint histogram to all zeros
    p = np.zeros(hist_size)

    for k in range(n):
        p[I[k], J[k]] = p[I[k], J[k]] + 1

    #------------------------------------------------------------------#
    # At this point, p contains the counts of cooccuring
    # intensities in the two images. You need to implement one final
    # step to make p take the form of a probability mass function
    # (p.m.f.).
    # Normalise the histogram in such a way that all values sum to 1
    p = p / p.sum();    # normalise p by the sum of all elements in p
    #------------------------------------------------------------------#

    return p


def mutual_information(p):
    # Compute the mutual information from a joint histogram.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)

    #------------------------------------------------------------------#
    # Implement the computation of the mutual information from p,
    # p_I and p_J. This can be done with a single line of code, but you
    # can use a for-loop instead.
    # HINT: p_I is a column-vector and p_J is a row-vector so their
    # product is a matrix. You can also use the sum() function here.
    MI = np.sum( p * np.log(p / (p_I * p_J)) )
    
    # Test: Shannon entropy (should be equal to MI if I=J)
    H = - np.sum(p_I * np.log(p_I) )
    #------------------------------------------------------------------#

    return MI #, H        # added H as output


def mutual_information_e(p):
    # Compute the mutual information from a joint histogram.
    # Alternative implementation via computation of entropy.
    # Input:
    # p - joint histogram
    # Output:
    # MI - mutual information in nat units
    # a very small positive number

    EPSILON = 10e-10

    # add a small positive number to the joint histogram to avoid
    # numerical problems (such as division by zero)
    p += EPSILON

    # we can compute the marginal histograms from the joint histogram
    p_I = np.sum(p, axis=1)
    p_I = p_I.reshape(-1, 1)
    p_J = np.sum(p, axis=0)
    p_J = p_J.reshape(1, -1)

    #------------------------------------------------------------------#
    # Implement the computation of the mutual information via
    # computation of entropy.
    H_I  = - np.sum(p_I * np.log(p_I))
    H_J  = - np.sum(p_J * np.log(p_J))
    H_IJ = - np.sum(p   * np.log(p))
    MI   = H_I + H_J - H_IJ
    #------------------------------------------------------------------#

    return MI


# SECTION 4. Towards intensity-based image registration


def ngradient(fun, x, h=1e-3):
    # Computes the derivative of a function with numerical differentiation.
    # Input:
    # fun - function for which the gradient is computed
    # x - vector of parameter values at which to compute the gradient
    # h - a small positive number used in the finite difference formula
    # Output:
    # g - vector of partial derivatives (gradient) of fun

    g = np.zeros_like(x)

    #------------------------------------------------------------------#
    # Implement the  computation of the partial derivatives of
    # the function at x with numerical differentiation.
    # g[k] should store the partial derivative w.r.t. the k-th parameter

    # g = ( fun(x+h/2) - fun(x-h/2) ) / h
    
    for k in range(x.size):
        xh1 = x.copy()
        xh2 = x.copy()
        xh1[k] = xh1[k] + h/2
        xh2[k] = xh2[k]- h/2
        a = fun(xh1)
        b = fun(xh2)
        g[k] = (a-b)/h
    #------------------------------------------------------------------#

    return g


def rigid_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-correlation between a fixed and
    # a moving image transformed with a rigid transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the rigid transform: the first element
    #     is the rotation angle and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    SCALING = 100

    # the first element is the rotation angle
    T = rotate(x[0])

    # the remaining two element are the translation
    tr = x[1:]    # added this line to make it clearer what happens at Th 
    # the gradient ascent/descent method work best when all parameters
    # of the function have approximately the same range of values
    # this is  not the case for the parameters of rigid registration
    # where the transformation matrix usually takes  much smaller
    # values compared to the translation vector this is why we pass a
    # scaled down version of the translation vector to this function
    # and then scale it up when computing the transformation matrix
    Th = util.t2h(T, tr*SCALING)          # changed x[1:] to tr

    # transform the moving image
    Im_t, Xt = image_transform(Im, Th)

    # compute the similarity between the fixed and transformed
    # moving image
    C = correlation(I, Im_t)

    if return_transform:
        return C, Im_t, Th
    else:
        return C


def affine_corr(I, Im, x, return_transform=True):
    # Computes normalized cross-corrleation between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the affine transform: the first element
    #     is the rotation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im)  
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)
    
    # NUM_BINS = 64
    # SCALING = 100 # ??????

    #------------------------------------------------------------------#
    # Implement the missing functionality
    # the first element is the rotation angle
    T_rot = rotate(x[0])
    # the second and third elements are the scaling parameters 
    T_sca = scale(x[1], x[2])
    # the fourth and fifth elements are the shearing parameters
    T_she = shear(x[3], x[4])
    # the sixth and seventh elements are the translation 
    t = x[5:7] 
    
    # Put the operations together to make the transformation matrix 
    T = T_rot.dot(T_sca).dot(T_she)

    # Make the homogeneous transformation matrix 
    Th = util.t2h(T, t)

    # Transform the moving image 
    Im_t, Xt = image_transform(Im, Th)
    
    # compute the similarity between the fixed and transformed
    # moving image
    C = correlation(I, Im_t)
    #------------------------------------------------------------------#

    if return_transform:
        return C, Im_t, Th
    else:
        return C


def affine_mi(I, Im, x, return_transform=True):
    # Computes mutual information between a fixed and
    # a moving image transformed with an affine transformation.
    # Input:
    # I - fixed image
    # Im - moving image
    # x - parameters of the affine transform: the first element
    #     is the rotation angle, the second and third are the
    #     scaling parameters, the fourth and fifth are the
    #     shearing parameters and the remaining two elements
    #     are the translation
    # return_transform: Flag for controlling the return values
    # Output:
    # C - normalized cross-correlation between I and T(Im) ------- THIS SEEMS COPIED FROM ABOVE BUT NOT WHAT WE WANT TO RETURN, CHANGING TO MI 
    # MI - mutual information between I and T(Im) 
    # Im_t - transformed moving image T(Im)
    # Th - transformation matrix (only returned if return_transform=True)

    NUM_BINS = 64
    SCALING = 100
    
    #------------------------------------------------------------------#
    # Implement the missing functionality
    # the first element is the rotation angle
    T_rot = rotate(x[0])
    # the second and third elements are the scaling parameters 
    T_sca = scale(x[1], x[2])
    # the fourth and fifth elements are the shearing parameters
    T_she = shear(x[3], x[4])
    # the sixth and seventh elements are the translation 
    t = x[5:7] 
    
    # Put the operations together to make the transformation matrix 
    T = T_rot.dot(T_sca).dot(T_she)

    # Make the homogeneous transformation matrix 
    Th = util.t2h(T, t)

    # Transform the moving image 
    Im_t, Xt = image_transform(Im, Th)
    
    # Compute the joint histogram between I and Im_t
    p = joint_histogram(I, Im_t, NUM_BINS)

    # Compute the mutual information between I and Im_t
    MI = mutual_information(p) 
    #------------------------------------------------------------------#

    # if return_transform:
    #     return C, Im_t, Th
    # else:
    #     return C

    if return_transform:
        return MI, Im_t, Th
    else:
        return MI
