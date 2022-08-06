import numpy as np
from skimage.filters import sobel_h, sobel_v
from skimage.segmentation import flood, flood_fill
import matplotlib.pyplot as plt
from flood_fill_tools import find_remeaning_components,get_centers
from skimage.morphology import skeletonize

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
        cv2.circle(rgb_img, e[0], 1, (255, 0, 255), -1)
        cv2.rectangle(rgb_img, e[1], e[2], (255, 0, 255), 2)

    for e in gen_cords:
        cv2.circle(rgb_img, (e[0], e[1]), int(e[2]*1.1), (0, 0, 255), 2)
        cv2.circle(rgb_img, (e[0], e[1]), 1, (0, 0, 255), -1)

    return rgb_img

def find_connections(img, start_point, end_point):
    # scale img to 0-1
    img = img.copy()
    img = img / 255

    img = skeletonize(img)
    img = img.astype(np.uint8)*255
    starty, startx  = start_point
    endy, endx = end_point
    offset = 2
    startx -= offset
    starty -= offset
    endx += offset
    endy += offset
    ends = []
    for i in range(startx, endx):
        if img[i, starty] == 255:
            ends.append((i, starty))
        if img[i, endy] == 255:
            ends.append((i, endy))
    for j in range(starty, endy):
        if img[startx, j] == 255:
            ends.append((startx, j))
        if img[endx, j] == 255:
            ends.append((endx, j))
    if len(ends) == 0:
        print("aaaaa")
        return
    if len(ends) != 2:
        print("Error: more than 2 ends found")
        return
    cv2.line(img, (ends[0][1], ends[0][0]), (ends[1][1], ends[1][0]), (255,255,255), 1)
    return img

def turn_off_components(img, resistor_cords, gen_cords):
    for e in resistor_cords:
        cv2.rectangle(img, e[1], e[2], (0,0,0), -1)
        img = find_connections(img, e[1], e[2])
    for e in gen_cords:
        r = int(e[2]*1.22)
        cv2.circle(img, (e[0], e[1]), r, (0,0,0), -1)
        img = find_connections(img, (e[0]-r, e[1]-r), (e[0]+r, e[1]+r))
    return img

def find_lines(img):
    img = img.copy()
    # find lines in image with HoughLines
    # lines = cv2.HoughLines(img, 1, np.pi / 180, 100, None, 0, 0)
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)
    if lines is None:
        print("No lines found")
        return img
    # convert lines to numpy array
    return lines

def visualize_lines(img, lines):
    # for i in range(0, len(lines)):
    #     rho = lines[i][0][0]
    #     theta = lines[i][0][1]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #     pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #     cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,255,0), 1)
    return img


def extract_junctions(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    centers = get_centers(img)
    return centers

def visualize_junctions(img, centers):
    for e in centers:
        cv2.circle(img, e, 1, (255, 0, 0), -1)
    return img


def cluster_symbols(img):
    # dilate img to get bigger symbols
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return img

def assign_details(details, gen_cords, resistor_cords, nodes):
    for r in resistor_cords:
        for n in nodes:
            if n[0] == r[0] and n[1] == r[1]:
                details.append((n, r))
                break