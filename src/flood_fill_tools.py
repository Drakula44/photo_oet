from matplotlib.pyplot import fill
import numpy as np
from skimage.segmentation import flood, flood_fill
import matplotlib.pyplot as plt
import cv2


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

def find_closest(matrix, target, radius):
    nonzero = np.argwhere(matrix != 0)
    distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
    a = np.where(distances < radius)
    print(a)
    nonzero = np.zeros(matrix.shape)
    nonzero[a] = 1

def split_circuit(img):
    width = len(img)
    height = len(img[0])
    original = img.copy()
    elements = []
    for i in range(width):
        for j in range(height):
            if img[i][j] != 0:
                filled = flood_fill(img, (i,j), 0)
                count = np.count_nonzero(cv2.absdiff(filled, img))
                elements.append([count, (i,j)])
                img = filled
    elements.sort(reverse=True)
    without = flood_fill(original, elements[0][1], 0)
    return original - without, without

def find_remeaning_components(img):
    width = len(img)
    height = len(img[0])
    original = img.copy()
    elements = []
    footprint = np.array([[0,1,0],[1,1,1],[0,1,0]])
    for i in range(width):
        for j in range(height):
            if img[i][j] != 0:
                filled = flood_fill(img, (i,j), 0, footprint=footprint)
                count = np.count_nonzero(cv2.absdiff(filled, img))
                elements.append([count, (i,j)])
                img = filled
    elements.sort(reverse=True)
    resistors = []
    for e in elements:
        B = np.argwhere(flood(original, e[1]))
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
        aspect_ratio = (ystop - ystart) / (xstop - xstart)
        if (aspect_ratio < 0.5 or aspect_ratio > 2) and e[0] > 100:
            center = (int((xstop + xstart) / 2), int((ystop + ystart) / 2))
            resistors.append([center, (xstart, ystart), (xstop, ystop)])
            cv2.rectangle(original, (xstart, ystart), (xstop, ystop), 125, -1)
    return resistors


def get_centers(img):
    centers = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] != 0:
                flood_img = flood(img, (i,j))
                mass_x, mass_y = np.where(flood_img > 0)
                # mass_x and mass_y are the list of x indices and y indices of mass pixels
                cent_y = int(np.average(mass_x))
                cent_x = int(np.average(mass_y))
                centers.append((cent_x, cent_y))
                flood_fill(img, (i,j), 0, in_place=True)
    return centers

def find_bounding_box_details(img):
    width = len(img)
    height = len(img[0])
    details = []
    for i in range(width):
        for j in range(height):
            if img[i][j] == 0:
                continue
            B = np.argwhere(flood(img, (i,j)))
            (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
            details.append([np.array([xstart, ystart]), np.array([xstop, ystop])])
            flood_fill(img, (i,j), 0, in_place=True)
    return details