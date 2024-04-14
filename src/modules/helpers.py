import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv



def imshow(img: np.ndarray, label='', cv_color=True) -> None:
    """
    Args:
        img (np.ndarray): Image to show.
        label (string, optional): The name of the image.
        cv_color (boolean, optional): If image was read by OpenCV (cv2.imread()), color scheme is BGR
            and then image should be displayed by RGB scheme (because of plt.imshow()). By default, True.
    """
    if cv_color:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    else:
        plt.imshow(img)


def resize_ratio(img: np.ndarray, fx=.5, fy=.5) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image to show.
        fx (floating, optional): Resize x-axis of image by given value. By default, 0.5.
        fy (floating, optional): Resize y-axis of image by given value. By default, 0.5.
    """
    return cv.resize(img, None, fx=fx, fy=fy)


def resize_tuple(img: np.ndarray, h=500, w=500) -> np.ndarray:
    """
    Args:
        img (np.ndarray): Image to show.
        h (integer, optional): Resize height of the image to given value. By default, 500.
        w (integer, optional): Resize width of the image to given value. By default, 500.
    """
    return cv.resize(img, (h, w))


if __name__ == '__main__':
    img = cv.imread('query_images/query_4.jpg')
    res_img = resize_tuple(img, 500, 500)
    imshow(img)
    imshow(res_img)
