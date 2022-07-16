import cv2
import numpy as np
import matplotlib.pyplot as plt

from template import Template

template_folder = "templates"

def get_connections_by_hand(gray):
    # threshold image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 150-161
    plt.imshow(thresh[:, 61:72])
    # plt.imshow(thresh)
    plt.show()
    print(np.shape(thresh))

