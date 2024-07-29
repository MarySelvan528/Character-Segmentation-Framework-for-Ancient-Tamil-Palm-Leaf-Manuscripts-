import numpy as np
# some useful functions...
from functools import reduce
import cv2 as cv


###################################################################
def derivate_image(im, angle):
    '''
    Compute derivative of input image
    :param im: input image. should be grayscale!
    :param angle: 0 or 90 degrees
    :return: computed derivative along that direction.
    includes padding...
    '''
    h, w = np.shape(im)
    pad_im = np.pad(im, (1, 0), 'edge')
    if angle == 'horizontal':  # horizontal derivative
        deriv_im = pad_im[1:, :w] - im  # [1:, :w]
    elif angle == 'vertical':
        deriv_im = pad_im[:h, 1:] - im  # [1:, :w]

    return deriv_im


###################################################################
def extract_ltrp1(im_d_x, im_d_y):
    """
    Extract LTrP1 code (4 orientations) by using input dx and dy matrices.
    ###################################################################
    Implemented by: Adrian Ungureanu - June 2019. If you use this script, please reference the source code (and send me
    an e-mail. I would be happy to hear about other people's success using my code (: )
    ###################################################################
    :param im_d_x: derivative of image according to x axis (horizontal)
    :param im_d_y: derivative of image according to y axis (vertical)
    :return: encoded LTrP1 code. Possible values ={1,2,3,4}
    """
    encoded_image = np.zeros(np.shape(im_d_y))  # define empty matrix, of the same shape as the image...

    # # apply conditions for each orientation...
    encoded_image[np.bitwise_and(im_d_x >= 0, im_d_y >= 0)] = 1
    encoded_image[np.bitwise_and(im_d_x < 0, im_d_y >= 0)] = 2
    encoded_image[np.bitwise_and(im_d_x < 0, im_d_y < 0)] = 3
    encoded_image[np.bitwise_and(im_d_x >= 0, im_d_y < 0)] = 4

    return encoded_image


###################################################################
def extract_ltrp2(ltrp1_code, im_side1, im_side2):
    """
    Extracting the P-components for every pixel (g_c), as defined in the original paper by S. Murala, R. P. Maheshwari
    and R. Balasubramanian (2012), "Local Tetra Patterns: A New Feature Descriptor for Content-Based Image Retrieval,"
    in IEEE Transactions on Image Processing, vol. 21, no. 5, pp. 2874-2886, May 2012. doi: 10.1109/TIP.2012.2188809.
    This implementation does not consider the MAGNITUDE, but that feature can be easily implemented...
    ###################################################################
    Implemented by: Adrian Ungureanu - June 2019. If you use this script, please reference the source code (and send me
    an e-mail. I would be happy to hear about other people's success using my code (: )
    ###################################################################
    :param ltrp1_code: previously computed LTrP1 code (with 4 possible orientations)
    :param plotting_flag: whether or not to display the P-components of the LTrP
    :return: the P-components stacked together. Output shape = (12, image_size, image_size)
    """
    this_im_side = np.shape(ltrp1_code)[0]
    ltrp1_code = np.pad(ltrp1_code, (1, 1), 'constant', constant_values=0)
    g_c1 = np.zeros((3, this_im_side, this_im_side))
    g_c2 = np.zeros((3, this_im_side, this_im_side))
    g_c3 = np.zeros((3, this_im_side, this_im_side))
    g_c4 = np.zeros((3, this_im_side, this_im_side))

    for i in range(1, im_side1 + 1):
        for j in range(1, im_side2 + 1):
            g_c = ltrp1_code[i, j]

            # # extract neighborhood around g_c pixel
            neighborhood = np.array([ltrp1_code[i + 1, j], ltrp1_code[i + 1, j - 11], ltrp1_code[i, j - 11],
                                     ltrp1_code[i - 1, j - 1], ltrp1_code[i - 1, j], ltrp1_code[i - 1, j + 1],
                                     ltrp1_code[i, j + 1], ltrp1_code[i + 1, j + 1]])
            # # determine the codes that are different from g_c
            mask = neighborhood != g_c
            # # apply mask
            ltrp2_local = np.multiply(neighborhood, mask)

            # # construct P-components for every orientation.
            if g_c == 1:
                for direction_index, direction in enumerate([2, 3, 4]):
                    g_dir = ltrp2_local == direction
                    g_c1[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=np.int))

            elif g_c == 2:
                for direction_index, direction in enumerate([1, 3, 4]):
                    g_dir = ltrp2_local == direction
                    g_c2[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=np.int))

            elif g_c == 3:
                for direction_index, direction in enumerate([1, 2, 4]):
                    g_dir = ltrp2_local == direction
                    g_c3[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=np.int))

            elif g_c == 4:
                for direction_index, direction in enumerate([1, 2, 3]):
                    g_dir = ltrp2_local == direction

                    g_c4[direction_index, i - 1, j - 1] = reduce(lambda a, b: 2 * a + b, np.array(g_dir, dtype=np.int))
                    pass

            elif g_c not in [1, 2, 3, 4]:
                raise Exception('Error - Invalid value for g_c. List of possible values include [1,2,3,4].')

    # # collect all P-components in a 'large_g_c'
    large_g_c = []
    for this_g_c in [g_c1, g_c2, g_c3, g_c4]:
        large_g_c.extend(this_g_c)
    large_g_c = np.array(large_g_c)

    return large_g_c


def LTRP_Pattern(image):
    deriv_h = derivate_image(im=image, angle='horizontal')
    deriv_v = derivate_image(im=image, angle='vertical')
    # # Extract LTrP1 code...
    ltrp1 = extract_ltrp1(im_d_x=deriv_h, im_d_y=deriv_v)
    # # Extract LTrP2 P-components, based on the previously obtained LTrP1
    im_side1 = image.shape[0]
    im_side2 = image.shape[1]
    ltrp2 = extract_ltrp2(ltrp1, im_side1, im_side2)
    # winname = 'im'
    # cv.namedWindow(winname)  # Create a named window
    # cv.moveWindow(winname, 40, 30)  # Move it to (40,30)
    # for i in range(12):
    #     cv.imshow(winname, ltrp2[i].astype('uint8')*255.0)
    #     cv.waitKey(0)
    # cv.imshow('im', ltrp1.astype('uint8')*255.0)
    # cv.waitKey(0)
    return ltrp2[0]
