from astropy.coordinates import Angle
import numpy as np
import re

"""
Provides convenience functions for astrometry
- convert_to_angle: convert input to astropy.coordinates.Angle
- common_size: returns the common size between a set of inputs.
- get_oneaxis: converts a string to a 3 D ndarray
- get_axis:  converts an array of strings to a (3, N) ndarray
"""


axismatch = re.compile('^([+-])([XYZ])$')


def convert_to_angle(data):
    """
    Convert input data to astropy.coordinates.Angle. Raise error if fails.

    Parameters
    ----------

    data: any input that can be converted to Angle.

    Returns
    -------
    astropy.coordinates.Angle

    """
    print("type(data) = {}".format(type(data)))
    if name is None:
        name = "input"
    try:
        wdata = Angle(data)
    except:
        raise ValueError("{} must be an angle or and angular quantity".format(data))
    return wdata


def common_size(input):
    """
    
    :return: 
    """
    all_lengths = []
    for elem in input:
        if elem is not None:
            if np.isscalar(elem):
                all_lengths.append(1)
            else:
                if isinstance(elem, np.ndarray):
                    all_lengths.append(elem.size)
                else:
                    all_lengths.append(len(elem))
        else:
            raise ValueError("Undefined input")
    return np.max(np.array(all_lengths))

def get_oneaxis(axis):
    """
    Convert a string such as '-Y' to a 3 dimension ndarray (in this case,
    [0, -1, 0])

    Parameters:
    -----------

    axis: string

    Return
    ------
    numpy.ndarray of shape (3,)

    """
    m = axismatch.match(axis)
    if m:
        if m.group(2) == 'X':
            result = np.array([1, 0, 0])
        elif m.group(2) == 'Y':
            result = np.array([0, 1, 0])
        elif m.group(2) == 'Z':
            result = np.array([0, 0, 1])
        else:
            raise ValueError("Invalid axis: {}".format(m.group(1)))
        if m.group(1) == '-':
            result = -result
    else:
        raise ValueError("Unrecognized axis: {}".format(axis))
    return result

def get_axis(axis):
    """
    Convert an array of N strings describing axes into a (3, N) numpy.ndarray

    Parameters
    ----------

    axis:  string or array of strings

    Returns
    -------

    np.ndarray((3, N))
    """
    if isinstance(axis, np.ndarray):
        result = np.zeros((3, len(axis)))
        for i, elem in enumerate(axis):
            result[:,i] = get_oneaxis(elem)
    else:
        result = get_oneaxis(axis)
    return result


