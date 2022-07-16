import cv2
from cv2 import resize
import numpy as np
import math
import os
import random  
import matplotlib.pyplot as plt
from templates.load_templates import load_templates


# read all template images from template folder
default_templates = load_templates("templates")
template_radius = 147

def get_radius(filename):
    # read image primer_kola
    img = cv2.imread(filename)
    # convert image to black and white
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find circles in image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=100, minRadius=0, maxRadius=0)
    radius = 0
    # draw circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            radius = i[2]
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    return radius

def get_coefs(line_ends):
    A = line_ends[1] - line_ends[3]
    B = line_ends[2] - line_ends[0]
    C = line_ends[0]*line_ends[3] - line_ends[2]*line_ends[1]
    return A, B, -C

def is_between(x, left, right):
    p = left
    left = min(left, right)
    right = max(p, right)
    left -= 1
    right += 1
    return x >= left and x <= right

def close_enough(point1, point2):
    return abs(point1[0]-point2[0]) <= 1 and abs(point1[1]-point2[1]) <= 1

def get_intersection(line1, line2):
    A1, B1, C1 = get_coefs(line1)
    A2, B2, C2 = get_coefs(line2)
    D = A1*B2 - A2*B1
    if D == 0:
        return None
    Dx = C1*B2 - C2*B1
    Dy = A1*C2 - C1*A2
    x = Dx / D
    y = Dy / D
    if not (is_between(x, line1[0], line1[2]) and is_between(x, line2[0], line2[2])):
        return None
    if not (is_between(y, line1[1], line1[3]) and is_between(y, line2[1], line2[3])):
        return None
    return x, y


filenames = ["samples/small_size_kolo.png"]
for filename in filenames:
    radius = get_radius(filename)
    # read image 
    img = cv2.imread(filename)
    # convert image to black and white
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize_factor =  radius / template_radius
    print(resize_factor)
    templates = [temp.scale(resize_factor) for temp in default_templates]
    # treshold resized image
    resized_img = gray
    # invert image
    resized_img = 255 - resized_img

    for j, template in enumerate(templates):
        rotations = 2
        if template.name == "strujni_gen":
            rotations = 4
        for i in range(rotations):
            w, h = template.image.shape
            res = cv2.matchTemplate(resized_img,template.image,cv2.TM_CCOEFF_NORMED,mask=template.mask)
            threshold = 0.99
            res[np.isnan(res)] = 0
            res[np.isinf(res)] = 0

            print(template.name, res.max())
            # plt.imshow(template.image)
            # plt.show()
            # plt.imshow(template.mask)
            # plt.show()
            plt.imshow(res)
            plt.show()
            # get 10 biggest elements from res
            print(np.sort(res.flatten())[::-1][:10])
            
            loc = np.where( res >= max(0.5, np.max(res.flatten()) * threshold))
            # show res
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)  
                cv2.rectangle(resized_img, pt, (pt[0] + w, pt[1] + h), (0,0,0), -1)
                # cv2.rectangle(resized_img, (pt[0]+template.connections[0][0][0],pt[1]+template.connections[0][0][1]), (pt[0]+template.connections[1][1][0],pt[1]+template.connections[1][1][1]), (0,0,0), -1)
            template.rotate_cw()

    plt.imshow(resized_img)
    plt.show()
    # cv2.imshow('asspenis', resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # invert colors
    # resized_img = cv2.bitwise_not(resized_img)

    rho = 1
    theta = np.pi / 180
    threshold = 100
    min_line_length = 40
    max_line_gap = 1
    lines = cv2.HoughLinesP(resized_img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(random.randint(0,255),random.randint(0,255),random.randint(0,255)),5)
    
    nodes = []
    for i in range(0,len(lines)):
        line1 = lines[i][0]
        for j in range(i, len(lines)):
            line2 = lines[j][0]
            node = get_intersection(line1, line2)
            if node is not None:
                nodes.append(node)

    for node in nodes:
        cv2.circle(img, (int(node[0]), int(node[1])), 5, (0,255,0), -1 )

    cv2.namedWindow('asspenis')
    cv2.moveWindow('asspenis', 50, 200)
    cv2.imshow('asspenis', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()