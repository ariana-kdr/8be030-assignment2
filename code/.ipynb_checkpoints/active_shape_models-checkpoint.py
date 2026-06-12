"""
Code for notebook active-shape-models
"""

import matplotlib.pyplot as plt
import numpy as np
import pca
import registration_util as reg_util
from IPython.display import clear_output, display

plt.rcParams["image.cmap"] = "gray"


def plot_hand_shapes():
    # Load the hand dataset (40 hand shapes)
    fn = "../data/dataset_hands/coordinates.txt"
    coordinates = np.loadtxt(fn)

    # Plotting a few shapes to see the variations
    fig = plt.figure(figsize=(16, 4))
    n = 0
    lbl = "hand_" + str(n + 1)
    ax1 = fig.add_subplot(141)
    ax1.plot(coordinates[n, :56], coordinates[n, 56:], label=lbl)
    ax1.set_title(lbl)
    n = 1
    lbl = "hand_" + str(n + 1)
    ax2 = fig.add_subplot(142)
    ax2.plot(coordinates[n, :56], coordinates[n, 56:], label=lbl)
    ax2.set_title(lbl)
    n = 2
    lbl = "hand_" + str(n + 1)
    ax3 = fig.add_subplot(143)
    ax3.plot(coordinates[n, :56], coordinates[n, 56:], label=lbl)
    ax3.set_title(lbl)
    n = 3
    lbl = "hand_" + str(n + 1)
    ax4 = fig.add_subplot(144)
    ax4.plot(coordinates[n, :56], coordinates[n, 56:], label=lbl)
    ax4.set_title(lbl)

    # ------------------------------------------------------------------#
    # TODO: Calculate the mean hand shape and plot this in a new figure.
    coord_avg = (coordinates.mean(0))
    lbl = "Mean hand shape" 
    fig2 = plt.figure(figsize=(16, 4))
    ax5 = fig2.add_subplot(144)
    ax5.plot(coordinates[1, :56], coordinates[1, 56:], label = lbl)
    ax5.set_title(lbl)

    # ------------------------------------------------------------------#


def pca_hands():
    fn = "../data/dataset_hands/coordinates.txt"
    coordinates = np.loadtxt(fn)
    # ------------------------------------------------------------------#
    # TODO: Apply PCA to the coordinates data.

    X_pca, v, w, fraction_variance = pca.mypca(coordinates)
    
    # Find how many dimensions are needed to describe 98% of variance
    num_dims = np.argmax(fraction_variance >= 0.98) + 1
    
    # Keep only the eigenvectors corresponding to these dimensions
    v_new = v[:, :num_dims]
    
    # ------------------------------------------------------------------#
    # Note: this function also needs to return the eigenvectors v and the
    # eigenvalues w (you will need these in the next exercise)
    
    return num_dims, v_new, v, w


def test_remaining_variance():
    # Load the data again and apply PCA
    fn = "../data/dataset_hands/coordinates.txt"
    coordinates = np.loadtxt(fn)
    mn = np.mean(coordinates, axis=0)
    num_dims, v_new, v, w = pca_hands()

    fig = plt.figure(figsize=(15, 10))

    # ------------------------------------------------------------------#
    # TODO: Create a loop to go through the dimensions left in v and
    # compute a variation that this dimension produces.

    # # Significant dimensions:
    # weight = 5
    
    # # Loop through the significant dimensions (those kept in v_new)
    # for i in np.arange(num_dims):

    #     dim = num_dims+1
    #     variation = weight * np.sqrt(w[dim])
        
    #     # Create shape at +5 std and -5 std along this eigenvector
    #     shape_plus  = mn + variation * v[:, i]
    #     shape_minus = mn - variation * v[:, i]
        
    #     # Plot both variations
    #     ax = fig.add_subplot(2, 3, i+1)
        
    #     # Reshape to x,y coordinates (coordinates are stored as [x1,x2,...,y1,y2,...])
    #     n_points = len(mn) // 2
    #     ax.scatter(shape_plus[:n_points],  shape_plus[n_points:],  color='b', s=20, label='+5std')
    #     ax.scatter(shape_minus[:n_points], shape_minus[n_points:], color='r', s=20, label='-5std')
    #     ax.plot(mn[:n_points], mn[n_points:], 'mediumseagreen', label='mean')
    #     ax.set_title('Dimension {}'.format(i+1))
    #     ax.legend()
    #     ax.axis('equal')

    # Remaining dimensions: 

    weight = 5
    
    remaining_dims = v.shape[1] - num_dims

    # find var + plot dimensions some dimensions
    for i in np.arange(9):
        dim = 3+num_dims
        variation = weight * np.sqrt(w[dim])
        shape_plus  = mn + variation * v[:, dim].real
        shape_minus = mn - variation * v[:, dim].real
        n_points = len(mn) // 2
  
        ax = fig.add_subplot(3, 3, i+1)  
        ax.scatter(shape_plus[:n_points],  shape_plus[n_points:],  color='r', s=10, label='+var')
        ax.scatter(shape_minus[:n_points], shape_minus[n_points:], color='b', s=10, label='-var')
        ax.plot(mn[:n_points], mn[n_points:], 'mediumseagreen', label='mean')
        ax.set_title('Dimension {}'.format(dim+i))
        ax.legend()
        ax.axis('equal')

    # ------------------------------------------------------------------#


def plot_hand_grayscale():
    fn = "../data/dataset_hands/test001.jpg"
    img_hand = plt.imread(fn)
    fn = "../data/dataset_hands/coordinates.txt"
    coordinates = np.loadtxt(fn)

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_hand)

    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    img2 = (
        img_hand[:, :, 0] * 299.0 / 1000
        + img_hand[:, :, 1] * 587.0 / 1000
        + img_hand[:, :, 2] * 114.0 / 1000
    )
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    # ------------------------------------------------------------------#
    # TODO: plot the hand template on top of the greyscale hand image
    # Compute mean shape from coordinates
    mn = np.mean(coordinates, axis=0) * 500
    n_points = len(mn) // 2
    
    # Plot mean hand template on top of grayscale image
    ax2.plot(mn[:n_points], mn[n_points:], label='template x 500')
    ax2.legend()
    # ------------------------------------------------------------------#


def test_transformed_hand():
    fn = "../data/dataset_hands/test001.jpg"
    img_hand = plt.imread(fn)
    fn = "../data/dataset_hands/coordinates.txt"
    coordinates = np.loadtxt(fn)
    mn = np.mean(coordinates, axis=0)

    # Initialize position
    # Convert mean shape to 2D format first (easier to work with for the next steps)
    initialpos = np.concatenate(
        (mn[:56].reshape(1, -1), mn[56:].reshape(1, -1)), axis=0
    )

    # ------------------------------------------------------------------#
    # TODO: Define a scaling/rotation/alignment matrix and transform the shape
    # as close as possible to the image (result: a variable called shape_t)
    
    # Define transformation params
    S = 500                 # scale
    angle = np.deg2rad(0)      # rot
    tx = 125                # trans x
    ty = -80                # trans y
    
    # Build rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    
    # Apply transformation: scale, rotate, translate
    shape_t = S * R.dot(initialpos) + np.array([[tx], [ty]])
    # ------------------------------------------------------------------#

    # Plot image and transformed shape
    fig = plt.figure(figsize=(16, 8))
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000
    img2 = (
        img_hand[:, :, 0] * 299.0 / 1000
        + img_hand[:, :, 1] * 587.0 / 1000
        + img_hand[:, :, 2] * 114.0 / 1000
    )
    ax2 = fig.add_subplot(122)
    ax2.imshow(img2, cmap="gray")
    ax2.plot(shape_t[0, :], shape_t[1, :], "r")
