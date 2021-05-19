import os
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R


def check_create(directory) :
    """
    Checks if a directory exists. If not, creates it.
    """
    
    existed = os.path.exists(directory)
    if not os.path.exists(directory) :
            os.mkdir(directory)
    return existed


def as_normrotvec(quat) :
    """
    Converts a quaternion to a normalized rotation vector. This way, only the information of the axis
    of rotation is kept.
    """
    
    rotvec = R(quat).as_rotvec()
    return rotvec/norm(rotvec)


def rotvec_cor(r1, r2) :
    """
    Computes the absolute correlation between two vectors.
    """
    return np.abs(r1.dot(r2))


def bin_cor(data, b) :
    """
    Computes the absolute correlation matrix of orientations of points cointained in a bin. This is done
    by converting the quaternions representing these rotations into normalized rotation vectors.
    
    :param data: dataset from which orientations are extracted
    :param b: numpy array which indexes points belonging to a certain bin
    """
    rotvec = [as_normrotvec(x) for x in data['orientations'][:][b]]
    len_bin = len(rotvec)
    cor = np.zeros((len_bin, len_bin))
    for i in range(len_bin) :
        for j in range(len_bin) :
            cor[i, j] = rotvec_cor(rotvec[i], rotvec[j])
    return cor


def find_center_axis(cor) :
    """
    Finds the center axis of a set of vectors as defined in notebook cdi_process.ipynb
    
    :param cor: correlation matrix used to find the center axis
    """
    metric = cor.sum(axis=1)
    return np.argmax(metric)
    
    
def similarity_score(cor) :
    """
    Computes the similarity score of a set of vectors as defined in notebook cdi_process.ipynb
    
    :param cor: correlation matrix representing relations in a set of vectors
    """
    
    return norm(cor)/len(cor)


def diff_angle(quat1, quat2) :
    """
    Computes the difference angle between two rotation represented as quaternions. This is 
    defined as the norm of the rotation vector representing the product of quat2 by quat1's 
    inverse rotation.
    """
    
    r_diff = (R(quat2) * R(quat1).inv()).as_rotvec()
    theta = norm(r_diff)
    if theta <= np.pi / 2 :
        return theta
    return theta - np.pi
    
    
def angle_2pts(ref, pt) :
    """
    Computes the angle between a point, a reference and the horizontal axis going through the
    reference in 2D. This is defined as the argument of the complex number representing the 
    difference of these two points.
    """
    
    delta = pt - ref
    z = complex(real = delta[0], imag=delta[1])
    return np.angle(z)/2


def angle_array(pts) :
    """
    Computes the angles between several 2D vectors and the mean of this set of vectors.
    
    :param pts: two-dimensional numpy array representing a set of 2D vectors
    """
    ref = pts.mean(axis=0)
    return np.array([angle_2pts(ref, pt) for pt in pts])