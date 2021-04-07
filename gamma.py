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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np

gamma_slider_max = 200
title_window = 'Gamma correction'


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    if rep == LOAD_GRAY_SCALE:
        img = cv2.imread(img_path, 2)
    else:  # rep = LOAD_RGB
        img = cv2.imread(img_path, 1)

    cv2.namedWindow(title_window)
    trackbar_name = 'Gamma %d' % gamma_slider_max
    cv2.createTrackbar(trackbar_name, title_window, 100, gamma_slider_max, trackBar)
    trackBar(0)
    print("Click on the screen to close the 'Gamma correction' window")
    cv2.waitKey()


def trackBar(n: int):
    gamma = float(n) / 100
    invGamma = 1000 if gamma == 0 else 1.0 / gamma
    max_ = 255
    gammaTable = np.array([((i / float(max_)) ** invGamma) * max_
                           for i in np.arange(0, max_ + 1)]).astype("uint8")
    img_ = cv2.LUT(img, gammaTable)
    cv2.imshow(title_window, img_)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
