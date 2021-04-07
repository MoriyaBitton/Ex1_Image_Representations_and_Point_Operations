"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 316451749


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if representation == 1:  # GRAY_SCALE
        imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imageGray = cv2.cvtColor(imageGray, cv2.COLOR_RGB2GRAY)
        return imageGray
    else:  # representation == 2 , RGB
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return imageRGB


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    YIQ_from_RGB = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])

    YIQ = imgRGB @ YIQ_from_RGB.transpose()
    return YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    YIQ_from_RGB = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    RGB_from_YIQ = np.linalg.inv(YIQ_from_RGB)  # invalid matrix
    RGB = imgYIQ @ RGB_from_YIQ.transpose()
    return RGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # change range of imgOrig from [0, 1] to [0, 255]
    norm_imgOrig = cv2.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_imgOrig = norm_imgOrig.astype(np.uint8)

    # Calculate the old image histogram (range = [0, 255])
    histOrg = np.zeros(256)
    for val in range(256):
        histOrg[val] = np.count_nonzero(norm_imgOrig == val)

    # Calculate the normalized Cumulative Sum (CumSum)
    cum_sum = np.cumsum(histOrg)

    # Create a LookUpTable(LUT)
    look_ut = np.floor((cum_sum / cum_sum.max()) * 255)

    # Replace each intensity i with LUT[i]
    imgEq = np.zeros_like(imgOrig, dtype=float)
    for i in range(256):
        imgEq[norm_imgOrig == i] = int(look_ut[i])

    # Calculate the new image histogram (range = [0, 255])
    histEQ = np.zeros(256)
    for val in range(256):
        histEQ[val] = np.count_nonzero(imgEq == val)

    # norm imgEQ from range [0, 255] to range [0, 1]
    imgEq = imgEq / 255.0

    return imgEq, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if len(imOrig.shape) == 2:  # single channel (grey channel)
        return oneChanelQuantize(imOrig.copy(), nQuant, nIter)

    # transfer img from RGB to YIQ
    # following quantization procedure should only operate on the Y channel
    yiqImg = transformRGB2YIQ(imOrig)
    qImage_, mse = oneChanelQuantize(yiqImg[:, :, 0].copy(), nQuant, nIter)  # y channel = yiqImg[:, :, 0].copy()
    qImage = []
    for img in qImage_:
        # convert the original img back from YIQ to RGB
        qImage_i = transformYIQ2RGB(np.dstack((img, yiqImg[:, :, 1], yiqImg[:, :, 2])))
        qImage.append(qImage_i)

    return qImage, mse


def oneChanelQuantize(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an single channel image (grey channel) in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # to return
    qImages = []
    error_i = []

    imOrig = cv2.normalize(imOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    imOrig_flat = imOrig.ravel().astype(int)
    histOrg, edges = np.histogram(imOrig_flat, bins=256)
    z_board = np.zeros(nQuant + 1, dtype=int)  # k + 1 boards

    # split the boarder to even parts.
    # the first board is 0 the last board is 255
    for i in range(nQuant + 1):
        z_board[i] = i * (255.0 / nQuant)

    # num of loops
    for i in range(nIter):
        # vector of weighted avg
        x_bar = []
        # calc mean weighted avg for every part
        for j in range(nQuant):
            intense = histOrg[z_board[j]:z_board[j + 1]]
            idx = range(len(intense))  # new arr in len intense
            weightedMean = (intense * idx).sum() / np.sum(intense)
            # add to x_bar mean between two boards
            x_bar.append(z_board[j] + weightedMean)

        qImage_i = np.zeros_like(imOrig)

        # overriding old color and update the mean color for every part
        # there is nQuant means
        for k in range(len(x_bar)):
            qImage_i[imOrig > z_board[k]] = x_bar[k]

        mse = np.sqrt((imOrig - qImage_i) ** 2).mean()
        error_i.append(mse)
        qImages.append(qImage_i / 255.0)  # back to range [0, 1]
        for k in range(len(x_bar) - 1):
            z_board[k + 1] = (x_bar[k] + x_bar[k + 1]) / 2  # move (k-2) middle boards -> b_i by x_bar's mean

    return qImages, error_i
