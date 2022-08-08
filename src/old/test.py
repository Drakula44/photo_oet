import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from skimage.morphology import skeletonize
from scipy.fft import fft2, fftshift
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.data import astronaut
from skimage.filters import window
from skimage import filters

from old.flood_fill_tools import flood_recursive, flood_recursive_find_max, find_closest
sys.setrecursionlimit(10000)


filename = "samples/small_size_kolo.png"
img = cv2.imread(filename)
color_img = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
pure_img = img.copy()

# threshold image
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
org = img.copy()

delta = org.copy()
flood_recursive(delta, flood_recursive_find_max(img))

img = delta.copy()
delta = org - delta
delta = delta//255
unique_values = np.unique(delta)

skeleton = skeletonize(delta)
skeleton = skeleton.astype(np.uint8)
skeleton *= 255

circles = cv2.HoughCircles(skeleton, cv2.HOUGH_GRADIENT, 1, 1, param1=1, param2=30, minRadius=0, maxRadius=100)
if circles is None:
    quit()
circles = np.uint16(np.around(circles))


skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
for i in circles[0, :]:
    cv2.circle(skeleton_rgb, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(skeleton_rgb, (i[0], i[1]), 2, (0, 0, 255), 3)



edge_sobel_h = filters.sobel_h(skeleton)
edge_sobel_v = filters.sobel_v(skeleton)
edge_sobel = np.abs(edge_sobel_h*edge_sobel_v)
edge_sobel = edge_sobel*255
edge_sobel = edge_sobel.astype(np.uint8)


kernel = np.ones((3, 3), np.uint8)
gray = cv2.dilate(edge_sobel, kernel, iterations=1)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 15, param1=1, param2=30, minRadius=0, maxRadius=100)
if circles is None:
    quit()


circles = np.uint16(np.around(circles))
gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
for i in circles[0, :]:
    cv2.circle(gray_rgb, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(gray_rgb, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.circle(gray, (i[0], i[1]), int(i[2]*1.5), (0, 0, 0), -1)
    find_closest(gray, (i[0], i[1]), int(i[2]*1.7))
