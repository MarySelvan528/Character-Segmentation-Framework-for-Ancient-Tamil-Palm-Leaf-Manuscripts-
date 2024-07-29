import numpy as np
import cv2 as cv


# https://www.geeksforgeeks.org/create-local-binary-pattern-of-an-image-using-opencv-python/
def LBP_Pixel(segment, centerx, centery):
    center = segment[centerx, centery]
    power_val = np.asarray([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    pixelPosition = np.asarray([
        [centerx - 1, centery - 1],     # Top Left
        [centerx - 1, centery],         # Top
        [centerx - 1, centery + 1],     # Top Right
        [centerx, centery + 1],         # Right
        [centerx + 1, centery + 1],     # Bottom Right
        [centerx + 1, centery],         # Bottom
        [centerx + 1, centery - 1],     # Bottom Left
        [centerx, centery - 1]          # Left
    ])
    values = segment[pixelPosition[:, 0], pixelPosition[:, 1]]
    values[values < center] = 0
    values[values >= center] = 1
    pixel = np.sum(values * power_val)
    return pixel


def LBP(image):
    grayImage = cv.cvtColor(image, cv.COLOR_RGB2GRAY) if len(image.shape) > 2 else image
    height, width = grayImage.shape
    img_lbp = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            xmin = i - 1 if i - 1 >= 0 else 0
            xmax = i + 2 if i + 2 <= grayImage.shape[0] else grayImage.shape[0]
            ymin = j - 1 if j - 1 >= 0 else 0
            ymax = j + 2 if j + 2 <= grayImage.shape[1] else grayImage.shape[1]
            segment = grayImage[xmin:xmax, ymin:ymax]
            centerx = abs(xmin - i)
            centery = abs(ymin - j)
            img_lbp[i, j] = LBP_Pixel(segment, centerx, centery)
    return img_lbp
