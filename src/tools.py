import numpy as np
from skimage.filters import sobel_h, sobel_v
from skimage.segmentation import flood, flood_fill
import matplotlib.pyplot as plt
from flood_fill_tools import find_remeaning_components
import cv2

def detect_circles(img):
    edge_sobel_h = sobel_h(img)
    edge_sobel_v = sobel_v(img)
    edge_sobel = np.abs(edge_sobel_h*edge_sobel_v)
    edge_sobel = edge_sobel*255
    edge_sobel = edge_sobel.astype(np.uint8)
    return edge_sobel

def fill_circles(img):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    # average = cv2.blur(dilated, (7, 7))
    # _, thresh = cv2.threshold(average, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # eroded = cv2.erode(thresh, kernel, iterations=1)
    return dilated

def get_gens_cords(img):
    img = img.copy()
    # find circles in image with HoughCircles
    maxRadius = int(max(img.shape[0], img.shape[1])/4)
    minRadius = maxRadius//8
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=minRadius, maxRadius=maxRadius)
    if circles is None:
        print("No generators found")
        return img
    # convert circles to numpy array
    circles = np.uint16(np.around(circles))
    return circles[0,:]


def turn_off_gens(img, circles):
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    for i in circles:
        cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 255), -1)
        img = flood_fill(img, (i[1], i[0]), 0)
    return img

def get_resistors_cords(img, circuit_graph_img):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(circuit_graph_img, kernel, iterations=3)
    circuit_graph_img = dilated
    img = img.copy()
    img = img * circuit_graph_img / 255

    return find_remeaning_components(img)


def visualize_components(rgb_img, gen_cords, resistor_cords):
    for e in resistor_cords:
        e = e[0]
        cv2.circle(rgb_img, (e[0], e[1]), 10, (255, 0, 255), -1)

    for e in gen_cords:
        cv2.circle(rgb_img, (e[0], e[1]), 10, (0, 0, 255), -1)
    
    return rgb_img