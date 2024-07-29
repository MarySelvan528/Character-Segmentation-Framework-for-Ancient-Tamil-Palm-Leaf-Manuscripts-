import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def Image_Results():
    Segment = []
    Images = np.load('Images.npy', allow_pickle=True)
    Image = Images[4]
    threshold = cv.threshold(Image, 75, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    analysis = cv.connectedComponentsWithStats(threshold, 4, cv.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    output = np.zeros(Image.shape, dtype="uint8")
    for j in range(1, totalLabels):
        area = values[j, cv.CC_STAT_AREA]
        componentMask = (label_ids == j).astype("uint8") * 255
        output = cv.bitwise_or(output, componentMask)
        index = np.where(componentMask != 0)
        x = np.min(index[0])
        y = np.min(index[1])
        w = np.max(index[0]) - x
        h = np.max(index[1]) - y
        image = np.zeros((w + 11, h + 11))
        image[index[0] - x + 5, index[1] - y + 5] = 255
        if 100 < area <= 1000:
            Segment.append(image)

    cv.imshow('Image', Image)
    cv.waitKey(0)

    fig, ax = plt.subplots(8, 8)
    for i in range(0, 64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(Segment[i])
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.tight_layout()
    path = "./Results/Image_1.png"
    plt.savefig(path)
    plt.show()

    fig, ax = plt.subplots(8, 8)
    for i in range(64, 128):
        plt.subplot(8, 8, i - 63)
        plt.imshow(Segment[i])
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.tight_layout()
    path = "./Results/Image_2.png"
    plt.savefig(path)
    plt.show()

    fig, ax = plt.subplots(8, 8)
    for i in range(128, len(Segment)):
        plt.subplot(8, 8, i - 127)
        plt.imshow(Segment[i])
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.tight_layout()
    path = "./Results/Image_3.png"
    plt.savefig(path)
    plt.show()

    # fig, ax = plt.subplots(8, 8)
    # for i in range(192, 256):
    #     plt.subplot(8, 8, i - 191)
    #     plt.imshow(Segment[i])
    # [axi.set_axis_off() for axi in ax.ravel()]
    # plt.tight_layout()
    # path = "./Results/Image_4.png"
    # plt.savefig(path)
    # plt.show()
    #
    # fig, ax = plt.subplots(5, 8)
    # for i in range(256, 295):
    #     plt.subplot(8, 8, i - 255)
    #     plt.imshow(Segment[i])
    # [axi.set_axis_off() for axi in ax.ravel()]
    # plt.tight_layout()
    # path = "./Results/Image_5.png"
    # plt.savefig(path)
    # plt.show()


def Enhancement():
    Images = np.load('Images.npy', allow_pickle=True)
    for i in range(Images.shape[0]):
        print(i)
        image = Images[i]
        alpha = 1.5  # Contrast control
        beta = 10  # Brightness control
        adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        Thresh = cv.threshold(adjusted, 75, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        # kernel = np.ones((3, 3), np.uint8)
        # opening = cv.morphologyEx(Thresh, cv.MORPH_OPEN, kernel, iterations=1)
        # closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=1)
        # Prep[closing == 255] = image[closing == 255]
        # Prep[Thresh == 255] = image[Thresh == 255]
        cv.imshow('Image', image)
        cv.imshow('Enhanced', Thresh)
        cv.waitKey(0)


def Img_Results():
    Images = np.load('Images.npy', allow_pickle=True)
    Preprocess = np.load('Preprocess.npy', allow_pickle=True)
    for i in range(len(Images)):
        cv.imwrite('./Results/Img_Results/image-%d.png' % (i + 1), Images[i])
        cv.imwrite('./Results/Img_Results/preprocess-%d.png' % (i + 1), np.uint8(Preprocess[i]))


if __name__ == '__main__':
    Img_Results()
    # Enhancement()
