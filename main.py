import cv2
import numpy as np
import math
import os
import random  
import matplotlib.pyplot as plt
needed_radius = 40

# read all template images from template folder
templates = []
template_folder = "templates"
template_filenames = os.listdir(template_folder)
template_filenames.sort()
for template_filename in template_filenames:
    # read template image
    template_img = cv2.imread(template_folder + "/" + template_filename)
    # convert template image to black and white
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    # treshold template image
    resize_factor = 40 / 44
    # resize image
    template_gray = cv2.resize(template_gray, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LANCZOS4)
    template_gray = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    

    templates.append(template_gray)


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
            # draw circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            print("radius:", i[2])
            # draw circle center
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    # # display img
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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

    resize_factor = needed_radius / radius
    print("resize_factor:", resize_factor)
    # resize image
    resized_img = cv2.resize(gray, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LANCZOS4)
    # treshold resized image
    resized_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    for j, template in enumerate(templates):
        for i in range(4):
            template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(resized_img,template,cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where( res >= threshold)
            plt.imshow(res)
            plt.show()
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)  
                cv2.rectangle(resized_img, pt, (pt[0] + w, pt[1] + h), (255,255,255), -1)
                if j == 2:
                    if w < h:
                        cv2.rectangle(resized_img, (pt[0],pt[1]+h//2-1), (pt[0] + w, pt[1]+h//2+1), (0,0,0), -1)
                    else:
                        cv2.rectangle(resized_img, (pt[0]+w//2-1,pt[1]), (pt[0]+w//2+1, pt[1]+h), (0,0,0), -1)
                else:
                    if w > h:
                        cv2.rectangle(resized_img, (pt[0],pt[1]+h//2-1), (pt[0] + w, pt[1]+h//2+1), (0,0,0), -1)
                    else:
                        cv2.rectangle(resized_img, (pt[0]+w//2-1,pt[1]), (pt[0]+w//2+1, pt[1]+h), (0,0,0), -1)
    cv2.imshow('asspenis', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # invert colors
    resized_img = cv2.bitwise_not(resized_img)

    # find all long lines in resized image
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 400  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(resized_img, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(random.randint(0,255),random.randint(0,255),random.randint(0,255)),5)
    # find all intesrections of lines

    """
    Mesto za Ognjena da nastavi da radi
    cava si
    imas ovaj lines vidis gore kako je formatiran tako da se snadji
    ostatak kola je relatino ok komentarisan ali naravno kucao sam ovo sat vremena tako da je sranje
    """

    

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

    print(len(nodes))

    cv2.namedWindow('asspenis')
    cv2.moveWindow('asspenis', 50, 200)
    cv2.imshow('asspenis', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()