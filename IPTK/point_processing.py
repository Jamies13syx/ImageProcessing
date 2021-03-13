import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt


def negative(file_path):
    try:
        original_arr = io.imread(file_path, as_gray=1)
    except ValueError:
        return None
    except OSError:
        return None
    negative_arr = 255 - original_arr
    return negative_arr


def intensity_level_slicing(file_path, bit_plane_level):
    try:
        original_arr = io.imread(file_path, as_gray=1)
    except ValueError:
        return None
    except OSError:
        return None
    mask = 2 ** (bit_plane_level - 1)
    intensity_level_slicing_arr = original_arr & mask
    return intensity_level_slicing_arr


def contrast_stretching(file_path):
    try:
        original_arr = io.imread(file_path, as_gray=1)
    except ValueError:
        return None
    except OSError:
        return None
    stretched_arr = np.ones(original_arr.shape, dtype="uint8")
    minValue = original_arr.min()

    maxValue = original_arr.max()
    for i in range(0, original_arr.shape[0]):
        for j in range(0, original_arr.shape[1]):
            if original_arr[i][j] == minValue:
                stretched_arr[i][j] = 0
            elif original_arr[i][j] == maxValue:
                stretched_arr[i][j] = 255
            else:
                stretched_arr[i][j] = round((original_arr[i][j] - minValue) * 255 / (maxValue - minValue))
    return stretched_arr


def power_law(file_path, power_law_constant, power_law_gamma):
    try:
        c = float(power_law_constant)
        gamma = float(power_law_gamma)
        if c <= 0:
            raise ValueError
    except ValueError:
        return None
    except OSError:
        return None

    try:
        original_arr = io.imread(file_path, as_gray=1)
    except ValueError:
        return None
    except OSError:
        return None
    gamma_arr = c * np.array(255 * (original_arr / 255) ** gamma, dtype='uint8')
    return gamma_arr


def counter(arr, mode, center):
    count = dict()
    if mode is 'global':
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                count[arr[i][j]] = count.get(arr[i][j], 0) + 1
    elif mode is 'local':
        for i in range(center[0] - 1, center[0] + 2):
            for j in range(center[1] - 1, center[1] + 2):
                count[arr[i][j]] = count.get(arr[i][j], 0) + 1
    return count


def global_histogram_equalization(file_path):
    try:
        original_arr = io.imread(file_path, as_gray=1)
    except ValueError:
        return None
    except OSError:
        return None
    count = counter(original_arr, 'global', (0, 0))
    equalized_arr = np.ones(original_arr.shape, dtype="uint8")
    for i in range(0, original_arr.shape[0]):
        for j in range(0, original_arr.shape[1]):
            sumValue = sum(count.get(k, 0) for k in range(0, original_arr[i][j] + 1))
            equalized_arr[i][j] = round(255 * sumValue/(original_arr.shape[0]*original_arr.shape[1]))
    return equalized_arr


def local_histogram_equalization(file_path):
    try:
        original_arr = io.imread(file_path, as_gray=1)
    except ValueError:
        return None
    except OSError:
        return None
    local_equalized_arr = np.ones(original_arr.shape, dtype="uint8")
    # equalized process
    for i in range(1, original_arr.shape[0] - 1):
        for j in range(1, original_arr.shape[1] - 1):
            count = counter(original_arr, 'local', (i, j))
            sumValue = sum(count.get(k, 0) for k in range(0, original_arr[i][j] + 1))
            local_equalized_arr[i][j] = round(255 * sumValue / 9)
    return local_equalized_arr
    

def histogram(file_path):
    try:
        original_arr = io.imread(file_path, as_gray=1)
    except ValueError:
        return -1
    except OSError:
        return -1

    plt.hist(original_arr.flatten(), bins=256)
    # make sub-figure's title
    plt.title("Histogram")
    plt.show()
    return 0


# if __name__ == '__main__':
#     img = Image.fromarray(np.uint8(power_law("F:/Neural Network/ComVision/images/DadAndSon.png", 1, 0.5)))
#     img.show()
