"""
Utility functions for segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import segmentation as seg
from scipy import ndimage
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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
    # TODO: Implement the  computation of the partial derivatives of
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

def scatter_data(X, Y, feature0=0, feature1=1, ax=None):
    # scater_data displays a scatterplot of at most 1000 samples from dataset X, and gives each point
    # a different color based on its label in Y

    k = 1000
    if len(X) > k:
        idx = np.random.randint(len(X), size=k)
        X = X[idx,:]
        Y = Y[idx]

    class_labels, indices1, indices2 = np.unique(Y, return_index=True, return_inverse=True)
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.grid()

    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == class_labels[i]
        # idx2 = indices2 == i
        lbl = 'X, class '+str(i)
        ax.scatter(X[idx2,feature0], X[idx2,feature1], color=c, label=lbl)

    return ax


def create_dataset(image_number, slice_number, task, include_edge=True):
    # create_dataset Creates a dataset for a particular subject (image), slice and task
    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task         - String corresponding to the task, either 'brain' or 'tissue'
    # include_edge - Decide if the Sobel edge detection gets included or not (included by default)
    # Output:
    # X           - Nxk feature matrix, where N is the number of pixels and k is the number of features
    # Y           - Nx1 vector with labels
    # feature_labels - kx1 cell array with descriptions of the k features

    #Extract features from the subject/slice
    X, feature_labels = extract_features(image_number, slice_number)

    if include_edge: 
        X_edge, feature_labels_edge = extract_features_edge(image_number, slice_number)
        X = np.concatenate([X, 
                            X_edge
                           ], axis=1)
        feature_labels = feature_labels + feature_labels_edge

    #Create labels
    Y = create_labels(image_number, slice_number, task)

    return X, Y, feature_labels

    

# Create some helper functions extract_features()
# Flatten 2D image to column vector
def flatten(img):
    img_flat = img.flatten().T.astype(float)
    img_flat = img.reshape(-1,1)
    return img_flat

# Compute local variance [ var(x) = E(x^2) + E(x)^2 ]
def local_var(img, size):
    mean = ndimage.uniform_filter(img, size=size)
    mean_squared = ndimage.uniform_filter(img**2, size=size)
    img_var = mean_squared - mean**2
    return img_var

# Compute sobel edge strength in x and y directions
def sobel_mag(img):
    img_sobel = np.sqrt(ndimage.sobel(img, axis=0)**2 + ndimage.sobel(img, axis=1)**2)
    return img_sobel

def extract_features(image_number, slice_number):
    # extracts features for [image_number]_[slice_number]_t1.tif and [image_number]_[slice_number]_t2.tif
    # Input:
    # image_number - Which subject (scalar)
    # slice_number - Which slice (scalar)
    # Output:
    # X           - N x k dataset, where N is the number of pixels and k is the total number of features
    # features    - k x 1 cell array describing each of the k features

    base_dir = '../data/dataset_brains/'

    t1 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t1.tif')
    t2 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t2.tif')


    #------------------------------------------------------------------#
    # TODO: Extract more features and add them to X.
    # Don't forget to provide (short) descriptions for the features

    # Sobel edge strength is removed from here because it is the extra feature we will add later 

    # Start assembling features 
    # Intensities 
    t1_intensity = flatten(t1)
    t2_intensity = flatten(t2) 

    # Gaussian smoothing
    t1_gaussian = flatten(ndimage.gaussian_filter(t1, sigma=20))
    t2_gaussian = flatten(ndimage.gaussian_filter(t2, sigma=20))

    # Texture differences using variance
    t1_var = flatten(local_var(t1, size=20))
    t2_var = flatten(local_var(t2, size=20))

    # Brightness differences in T1 vs T2
    t1t2_diff = flatten(t1 - t2)

    # Assemble features + labels
    X = np.concatenate([
        t1_intensity, 
        t2_intensity, 
        t1_gaussian,
        t2_gaussian, 
        t1_var,
        t2_var, 
        t1t2_diff
    ], axis=1)

    features = (
        'T1 intensity',
        'T2 intensity', 
        'T1 Gaussian, σ=20', 
        'T2 Gaussian, σ=20',
        'T1 local variance', 
        'T2 local variance', 
        'T1 - T2 difference'
    )

    #------------------------------------------------------------------#
    return X, features


def extract_features_edge(image_number, slice_number):
    # Extracts Sobel edge strength features for T1 and T2 images
    # This feature finds local intensity changes, which correspond to
    # tissue boundaries in MRI. It is different from the baseline features
    # bc measures rate of change of intensity instead of intensity itself 
    # or local averages.
    # Input:
    #   image_number - Which subject (scalar)
    #   slice_number - Which slice (scalar)
    # Output:
    #   X_edge   - N x 2 matrix with T1 and T2 Sobel edge strength per pixel
    #   features_edge - tuple of 2 feature description strings

    base_dir = '../data/dataset_brains/'
    t1 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t1.tif').astype(float)
    t2 = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_t2.tif').astype(float)

    # Boundary changes using Sobel edge detection
    t1_edge = flatten(sobel_mag(t1))
    t2_edge = flatten(sobel_mag(t2))

    # Assemble features + labels
    X_edge = np.concatenate([
        t1_edge,
        t2_edge,
    ], axis=1)

    features_edge = (
        'T1 Sobel edge strength',
        'T2 Sobel edge strength',
    )

    return X_edge, features_edge


def create_labels(image_number, slice_number, task):
    # Creates labels for a particular subject (image), slice and
    # task
    #
    # Input:
    # image_number - Number of the subject (scalar)
    # slice_number - Number of the slice (scalar)
    # task        - String corresponding to the task, either 'brain' or 'tissue'
    #
    # Output:
    # Y           - Nx1 vector with labels
    #
    # Original labels reference:
    # 0 background
    # 1 cerebellum
    # 2 white matter hyperintensities/lesions
    # 3 basal ganglia and thalami
    # 4 ventricles
    # 5 white matter
    # 6 brainstem
    # 7 cortical grey matter
    # 8 cerebrospinal fluid in the extracerebral space

    #Read the ground-truth image
    base_dir = '../data/dataset_brains/'

    I = plt.imread(base_dir + str(image_number) + '_' + str(slice_number) + '_gt.tif')

    if task == 'brain':
        Y = I>0
    elif task == 'tissue':
        # sub-binarize
        white_matter = np.isin(I, [2, 5])
        gray_matter = np.isin(I, [3, 7])
        csf = np.isin(I, [4, 8])
        background = np.isin(I, [0, 1, 6])

        # new GT
        Y = np.copy(I)
        Y[background] = 0
        Y[white_matter] = 1
        Y[gray_matter] = 2
        Y[csf] = 3
    else:
        print(task)
        raise ValueError("Variable 'task' must be one of two values: 'brain' or 'tissue'")

    Y = Y.flatten().T
    Y = Y.reshape(-1,1)

    return Y


def dice_overlap(true_labels, predicted_labels, smooth=1.):
    # returns the Dice coefficient for two binary label vectors
    # Input:
    # true_labels         Nx1 binary vector with the true labels
    # predicted_labels    Nx1 binary vector with the predicted labels
    # smooth              smoothing factor that prevents division by zero
    # Output:
    # dice          Dice coefficient

    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    #------------------------------------------------------------------#
    # TODO: Implement the missing functionality for Dice overlap
    # intersection: pixels that are +ve in both
    intersection = np.sum(t * p)
    
    # dice formula w smoothing -> prevent division by zero
    dice = (2. * intersection + smooth) / (np.sum(t) + np.sum(p) + smooth)
    #------------------------------------------------------------------#
    return dice


def dice_multiclass(true_labels, predicted_labels):
    #dice_multiclass.m returns the Dice coefficient for two label vectors with
    #multiple classses
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # dice_score          Dice coefficient

    all_classes, indices1, indices2 = np.unique(true_labels, return_index=True, return_inverse=True)

    dice_score = np.empty((len(all_classes), 1))
    dice_score[:] = np.nan

    #Consider each class as the foreground class
    for i in np.arange(len(all_classes)):
        idx2 = indices2 == all_classes[i]
        lbl = 'X, class '+ str(all_classes[i])
        temp_true = true_labels.copy()
        temp_true[true_labels == all_classes[i]] = 1  #Class i is foreground
        temp_true[true_labels != all_classes[i]] = 0  #Everything else is background

        temp_predicted = predicted_labels.copy();
        # print(temp_predicted.dtype)
        temp_predicted[predicted_labels == all_classes[i]] = 1
        temp_predicted[predicted_labels != all_classes[i]] = 0
        dice_score[i] = dice_overlap(temp_true.astype(int), temp_predicted.astype(int))

    dice_score_mean = dice_score.mean()

    return dice_score_mean


def classification_error(true_labels, predicted_labels):
    # classification_error.m returns the classification error for two vectors
    # with labels
    #
    # Input:
    # true_labels         Nx1 vector with the true labels
    # predicted_labels    Nx1 vector with the predicted labels
    #
    # Output:
    # error         Classification error

    assert true_labels.shape[0] == predicted_labels.shape[0], "Number of labels do not match"

    t = true_labels.flatten()
    p = predicted_labels.flatten()

    #------------------------------------------------------------------#
    # TODO: Implement the missing functionality for classification error
    err = np.sum(t != p) / len(t)
    #------------------------------------------------------------------#
    return err


    # NEW FUNCTIONS FOR ASSIGNMENT

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Inputs: 
    #     model
    #     X_train
    #     y_train
    #     X_test
    #     y_test
    # Outputs:
    #    err
    #    dice 
    #    y_pred 
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    err = classification_error(y_test, y_pred)
    dice = dice_multiclass(y_test.reshape(-1,1), y_pred.reshape(-1,1))
    
    return err, dice, y_pred




def show_segmentation(y_true, y_pred, title='Segmentation result'):
    # Inputs: 
    #     y_true
    #     y_pred
    #     title
 
    true_mask = y_true.reshape(240, 240)
    pred_mask = y_pred.reshape(240, 240)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(true_mask)
    axes[0].set_title('True mask')
    axes[0].axis('off')
    axes[1].imshow(pred_mask)
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()



def find_best_segmentation(train_subject, test_subject, slice_number=1, n_train_samples=10000):
    # Input:
    #    train_subject
    #    test_subject
    #    slice_number 
    #    n_train_subject 
    # Output:

    task = 'tissue' # could make this an input as well 
    
    # Load data for both subjects
    # Don't forget to set include_edge=True
    X_train, y_train, feature_labels = create_dataset(train_subject, slice_number, task, include_edge=True)
    X_test, y_test, _ = create_dataset(test_subject, slice_number, task, include_edge=True)
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Define feature sets to compare
    feature_sets = {
        'Intenisities only':     [0, 1],
        'Base features':         [0, 1, 2, 3, 4, 5, 6],
        'Base features + edges': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }

    # Define models to compare 
    models = {
    # 'kNN k=3': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
    'kNN k=7': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    # 'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'SVM (RBF kernel)':   make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=0, cache_size=500)),
    'Random Forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=-1)
    }

    # Subsample training pixels so notebook runs faster 
    N = min(n_train_samples, len(y_train))
    idx_train = np.random.choice(len(y_train), size=N, replace=False)

    # Run all combinations
    results = []
    for fs_name, cols in feature_sets.items():
        for model_name, model in models.items():
            err, dice, _ = evaluate_model(
                model,
                X_train[idx_train][:, cols], y_train[idx_train],
                X_test[:, cols], y_test
            )
            results.append((fs_name, model_name, err, dice))

    # Sort by Dice score + print
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
    print(f'\nResults for train subject {train_subject}, test subject {test_subject}, slice {slice_number}')
    print('Feature set | Model | Error | Dice')
    for fs_name, model_name, err, dice in results_sorted:
        print(f'{fs_name:35s} | {model_name:15s} | {err:.4f} | {dice:.4f}')

    # Extract + print best ones 
    best_fs, best_model, best_err, best_dice = results_sorted[0]
    print('Best choice:')
    print('Features:', best_fs)
    print('Model:', best_model)
    print('Error:', best_err)
    print('Dice:', best_dice)

    return best_fs, best_model, best_err, best_dice, results_sorted




    
    # n = t1.shape[0]
    # features = ()

    # t1f = t1.flatten().T.astype(float)
    # t1f = t1f.reshape(-1, 1)
    # t2f = t2.flatten().T.astype(float)
    # t2f = t2f.reshape(-1, 1)

    # X = np.concatenate((t1f, t2f), axis=1)

    # features += ('T1 intensity',)
    # features += ('T2 intensity',)