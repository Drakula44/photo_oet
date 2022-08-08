import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import Output


images = [cv2.imread('images/details/' + str(i) + '.png') for i in range(19)]

# images = [cv2.imread('images/details/' + str(3) + '.png')]


def detect_cross(img):
    # do a fourier transform on the image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    # plt.imshow(img, cmap='gray')
    # plt.show()


def process_detail(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if detect_cross(gray):
        return 'cross'
    # data = pytesseract.image_to_string(gray, config='--psm 6 --oem 1 -c tessedit_char_whitelist=VRgI0123456789')
    d = pytesseract.image_to_data(img, output_type=Output.DICT, config='--psm 6 --oem 1 -c tessedit_char_whitelist=EVRgI0123456789')
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if d['text'][i] == '':
            continue
        print(d['text'][i])
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return 

for i in images:
    process_detail(i)
    # plt.imshow(i)
    # plt.show()
    