import cv2
import pytesseract

an = 1
if an == 1:
    image = cv2.imread('1.jpg')
    image = cv2.resize(image, [512, 512])

    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    res, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    dilated = cv2.dilate(thresh, kernel, iterations=5)

    val, contours =cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    coord = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if h > 300 and w > 300:
            continue
        if h < 40 or w < 40:
            continue
        coord.append((x, y, w, h))

    coord.sort(key=lambda tup: tup[0])  # if the image has only one sentence sort in one axis

    count = 0
    for cor in coord:
        [x, y, w, h] = cor
        t = image[y:y + h, x:x + w, :]
        cv2.imwrite(str(count) + ".png", t)
    print("number of char in image:", count)