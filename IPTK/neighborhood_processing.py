import skimage.io as io
import numpy as np


def box_hotspot_1D(arr, hotspot, pad_size, weight, direction):
    result = 0
    if direction == 0:
        for i in range(-pad_size, pad_size + 1):  # only row
            if hotspot[0] - i < 0 or hotspot[0] - i >= arr.shape[0]:
                continue  # zero padding
            result += arr[hotspot[0] - i][hotspot[1]]
    else:
        for i in range(-pad_size, pad_size + 1):  # only column
            if hotspot[1] - i < 0 or hotspot[1] - i >= arr.shape[0]:
                continue  # zero padding, no contribution to result if it is zero
            result += arr[hotspot[0]][hotspot[1] - i]
    result = int(result / weight)
    return result


def box_smoothing_1D(arr, kernel):
    pad_size = int((max(kernel.shape[0], kernel.shape[1]) - 1) / 2)
    if kernel.shape[0] == 1:
        direction = 0
    else:
        direction = 1
    weight = kernel.shape[0] * kernel.shape[1]
    result_arr = np.zeros(arr.shape, dtype="int")
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            result_arr[i][j] = box_hotspot_1D(arr, (i, j), pad_size, weight, direction)
    return result_arr


def box_smoothing(file_path, k):
    try:
        original_arr = io.imread(file_path, as_gray=1)
        kernel_size = int(k)
    except ValueError:
        return None
    except OSError:
        return None
    kernel_1 = np.ones((kernel_size, 1), dtype="int")
    intermediate_arr = box_smoothing_1D(original_arr, kernel_1)
    kernel_2 = np.ones((kernel_size, 1), dtype="int")
    box_smoothing_arr = box_smoothing_1D(intermediate_arr, kernel_2)
    return box_smoothing_arr


def gaussian_hotspot_1D(arr, kernel, hotspot, pad_size, weight, direction):
    result = 0
    if direction == 0:
        for i in range(-pad_size, pad_size + 1):  # only column
            if hotspot[0] - i < 0 or hotspot[0] - i >= arr.shape[0]:
                continue  # zero padding
            result += arr[hotspot[0] - i][hotspot[1]] * kernel[0][i + pad_size]
    else:
        for i in range(-pad_size, pad_size + 1):  # only row
            if hotspot[1] - i < 0 or hotspot[1] - i >= arr.shape[0]:
                continue  # zero padding, no contribution to result if it is zero
            result += arr[hotspot[0]][hotspot[1] - i] * kernel[i + pad_size]

    return result / weight


def gaussian_smoothing_1D(arr, kernel):
    pad_size = int((max(kernel.shape[0], kernel.shape[1]) - 1) / 2)
    weight = 0
    if kernel.shape[0] == 1:
        direction = 0
        for i in range(0, kernel.shape[1]):
            weight += kernel[0][i]
    else:
        direction = 1
        for i in range(0, kernel.shape[0]):
            weight += kernel[i]
    result_arr = np.zeros(arr.shape, dtype="float")
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            result_arr[i][j] = gaussian_hotspot_1D(arr, kernel, (i, j), pad_size, weight, direction)
    return result_arr


def gaussian_smoothing(file_path, input_k, k, input_sigma):
    try:
        original_arr = io.imread(file_path, as_gray=1)
        constant_k = int(input_k)
        kernel_size = int(k)
        sigma = int(input_sigma)
        if kernel_size <= 0 or sigma <= 0 or constant_k <= 0:
            return None
    except ValueError:
        return None
    except OSError:
        return None
    mid = int(kernel_size / 2)
    gaussian_kernel_1 = np.ones((kernel_size, 1), dtype="float")
    for i in range(0, gaussian_kernel_1.shape[0]):
        gaussian_kernel_1[i] = constant_k * np.exp(- ((i - mid) ** 2) / (2 * sigma ** 2))
    intermediate_arr = gaussian_smoothing_1D(original_arr, gaussian_kernel_1)
    gaussian_kernel_2 = np.ones((1, kernel_size), dtype="float")
    for j in range(0, gaussian_kernel_2.shape[1]):
        gaussian_kernel_2[0][j] = constant_k * np.exp(- ((j - mid) ** 2) / (2 * sigma ** 2))
    gaussian_smoothing_arr = gaussian_smoothing_1D(intermediate_arr, gaussian_kernel_2)
    return gaussian_smoothing_arr


def laplacian_hotspot_2D(arr, kernel, hotspot, pad_size):
    result = 0
    for i in range(-pad_size, pad_size + 1):  # on each row
        if hotspot[0] - i < 0 or hotspot[0] - i >= arr.shape[0]:
            continue  # zero padding, no contribution to result if it is zero
        for j in range(-pad_size, pad_size + 1):  # on each column
            if hotspot[1] - j < 0 or hotspot[1] - j >= arr.shape[1]:
                continue
            result += arr[hotspot[0] - i][hotspot[1] - j] * kernel[i + pad_size][j + pad_size]
    return result


def laplacian_sharping_2D(arr, kernel):
    pad_size = int((kernel.shape[1] - 1) / 2)
    result_arr = np.zeros(arr.shape, dtype="int")
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            result_arr[i][j] = laplacian_hotspot_2D(arr, kernel, (i, j), pad_size)
    return result_arr


def get_laplacian_mask(original_arr):
    laplacian_kernel = np.ones((3, 3), dtype="int")
    laplacian_kernel[1][1] = -8
    laplacian_mask = laplacian_sharping_2D(original_arr, laplacian_kernel)
    minVal = laplacian_mask.min()
    maxVal = laplacian_mask.max()
    for i in range(0, laplacian_mask.shape[0]):
        for j in range(0, laplacian_mask.shape[1]):
            if laplacian_mask[i][j] == minVal:
                laplacian_mask[i][j] = 0
            elif laplacian_mask[i][j] == maxVal:
                laplacian_mask[i][j] = 255
            else:
                laplacian_mask[i][j] = int(255 * laplacian_mask[i][j] / (maxVal - minVal))
    return laplacian_mask


def laplacian_sharping(file_path):
    try:
        original_arr = io.imread(file_path, as_gray=1)
    except ValueError:
        return None, None
    except OSError:
        return None, None
    laplacian_mask = get_laplacian_mask(original_arr)
    laplacian_sharping_arr = original_arr - laplacian_mask
    return laplacian_sharping_arr, laplacian_mask


def median_hotspot_2D(arr, hotspot, pad_size):
    candidates = []
    for i in range(-pad_size, pad_size + 1):  # on each row
        for j in range(-pad_size, pad_size + 1):  # on each column
            if 0 <= hotspot[0] - i < arr.shape[0] and 0 <= hotspot[1] - j < arr.shape[1]:
                candidates.append(arr[hotspot[0] - i][hotspot[1] - j])
            else:
                candidates.append(0)
    candidates.sort()
    # print(candidates)
    return candidates[int(len(candidates) / 2)]


def median_filtering(arr, kernel_size):
    result_arr = np.zeros(arr.shape, dtype="int")
    pad_size = int((kernel_size - 1) / 2)
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            result_arr[i][j] = median_hotspot_2D(arr, (i, j), pad_size)
    return result_arr


def order_statistic(file_path, k):
    try:
        original_arr = io.imread(file_path, as_gray=1)
        kernel_size = int(k)
        if kernel_size <= 0:
            return None
    except ValueError:
        return None
    except OSError:
        return None
    result_arr = median_filtering(original_arr, kernel_size)
    return result_arr


def highboost(file_path, input_k, k, input_sigma):
    try:
        original_arr = io.imread(file_path, as_gray=1)
        constant_k = int(input_k)
        kernel_size = int(k)
        sigma = int(input_sigma)
        if kernel_size <= 0 or sigma <= 0 or constant_k < 0:
            return None
    except ValueError:
        return None
    except OSError:
        return None
    gaussian_smoothing_arr = gaussian_smoothing(file_path, 1, kernel_size, sigma)
    highboost_mask = original_arr - gaussian_smoothing_arr
    highboost_arr = original_arr + constant_k * highboost_mask
    return highboost_arr, highboost_mask
