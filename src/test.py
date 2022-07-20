import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import pytesseract
from skimage.morphology import skeletonize
from scipy.fft import fft2, fftshift
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.data import astronaut
from skimage.filters import window

sys.setrecursionlimit(100000)
filename = "samples/small_size_kolo.png"
img = cv2.imread(filename)
# invert image
color_img = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
pure_img = img.copy()

# threshold image
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
org = img.copy()

plt.imshow(img, cmap='gray')
plt.show()

def flood_recursive_find_max(matrix):
    width = len(matrix)
    height = len(matrix[0])
    max_count = 0
    max_cord = (0, 0)
    global count
    count = 0
    def fill(x,y,start_color,color_to_update):
        global count
        #if the square is not the same color as the starting point
        if matrix[x][y] != start_color:
            return
        #if the square is not the new color
        elif matrix[x][y] == color_to_update:
            return
        else:
            #update the color of the current square to the replacement color
            matrix[x][y] = color_to_update
            count += 1
            neighbors = [(x-1,y),(x+1,y),(x-1,y-1),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x,y-1),(x,y+1)]
            for n in neighbors:
                if 0 <= n[0] <= width-1 and 0 <= n[1] <= height-1:
                    fill(n[0],n[1],start_color,color_to_update)

    for i in range(width):
        for j in range(height):
            if matrix[i][j] != 0:
                start_color = matrix[i][j]
                fill(i,j,start_color,0)
                if count > max_count:
                    max_count = count
                    max_cord = (i,j)
                count = 0
    print(max_count, max_cord)
    return max_cord

def flood_recursive(matrix, cord):
    width = len(matrix)
    height = len(matrix[0])
    def fill(x,y,start_color,color_to_update):
        global count
        #if the square is not the same color as the starting point
        if matrix[x][y] != start_color:
            return
        #if the square is not the new color
        elif matrix[x][y] == color_to_update:
            return
        else:
            #update the color of the current square to the replacement color
            matrix[x][y] = color_to_update
            count += 1
            neighbors = [(x-1,y),(x+1,y),(x-1,y-1),(x+1,y+1),(x-1,y+1),(x+1,y-1),(x,y-1),(x,y+1)]
            for n in neighbors:
                if 0 <= n[0] <= width-1 and 0 <= n[1] <= height-1:
                    fill(n[0],n[1],start_color,color_to_update)

    i = cord[0]
    j = cord[1]
    start_color = matrix[i][j]
    fill(i,j,start_color,0)
    return matrix

delta = org.copy()
flood_recursive(delta, flood_recursive_find_max(img))
# plt.imshow(delta, cmap="gray")
# plt.show()
img = delta.copy()

delta = org - delta

plt.imshow(delta, cmap="gray")
plt.show()
delta = delta//255
# get all unique values from matrix
unique_values = np.unique(delta)
print(unique_values)
print(delta)
# skeletonize image
skeleton = skeletonize(delta)
plt.imshow(skeleton, cmap="gray")
plt.show()




image_f = np.abs(fftshift(fft2(skeleton)))

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
ax = axes.ravel()
ax[0].set_title("Original image")
ax[0].imshow(skeleton, cmap='gray')
ax[1].set_title("Original FFT (frequency)")
ax[1].imshow(np.log(image_f), cmap='magma')
plt.show()