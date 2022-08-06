import numpy as np
import matplotlib.pyplot as plt
import cv2
import random  

from image_processing import process_image

filename = "samples/small_size_kolo.png"
rgb_img = cv2.imread(filename)

gen_cords, resistor_cords, nodes, lines = process_image(rgb_img.copy())


def close_enoug_to_line(line, node):
    d1 = np.linalg.norm(line[0]-node)
    d2 = np.linalg.norm(line[1]-node)
    r = np.linalg.norm(line[1]-line[0])
    d= np.linalg.norm(np.cross(line[0]-line[1], node-line[1]))/np.linalg.norm(line[0]-line[1])
    return d < 3 and d1/r < 0.8 and d2/r < 0.8

def split_lines(lines, nodes):
    lines_split = []
    for line in lines:
        one_split = False
        for node in nodes:
            if close_enoug_to_line(line, node):
                one_split = True
                lines_split.append([line[0],node])
                lines_split.append([node,line[1]])
        if not one_split:
            lines_split.append(line)
    return lines_split

def close_enough(p1, p2):
    return np.linalg.norm(p1-p2) < 3

def merge_point(lines):
    for l1 in range(len(lines)):
        for l2 in range(len(lines)):
            if close_enough(lines[l1][0], lines[l2][0]):
                lines[l2][0] = lines[l1][0]
            if close_enough(lines[l1][0], lines[l2][1]):
                lines[l2][1] = lines[l1][0]
            if close_enough(lines[l1][1], lines[l2][0]):
                lines[l2][0] = lines[l1][1]
            if close_enough(lines[l1][1], lines[l2][1]):
                lines[l2][1] = lines[l1][1]
    return lines

def add_components(lines, nodes, components):
    splited_lines = []
    for line in lines:
        one_split = False
        for component in components:
            if close_enoug_to_line(line, component[0]):
                one_split = True
                splited_lines.append([line[0],component[0]])
                splited_lines.append([component[0],line[1]])
                nodes.append(component)
        if not one_split:
            splited_lines.append(line)
    return splited_lines, nodes


# remove radius from coordinates




lines = split_lines(lines, nodes)
lines = merge_point(lines)
img = rgb_img.copy()
for l in lines:
    l = (list(l[0]),list(l[1].astype(int)))
    cv2.line(img, l[0],l[1], (random.randint(0,255),random.randint(0,255),random.randint(0,255)),5)


nodes = np.unique(np.array([l[0] for l in lines] + [l[1] for l in lines]),axis=0)
print(nodes)

nodes = [[i, "N"] for i in nodes]

lines, nodes = add_components(lines, nodes, resistor_cords)

img = rgb_img.copy()
for l in lines:
    l = (list(l[0]),list(l[1].astype(int)))
    print(l)
    cv2.line(img, l[0],l[1], (random.randint(0,255),random.randint(0,255),random.randint(0,255)),5)


lines, nodes = add_components(lines, nodes, gen_cords)


for l in lines:
    l = (list(l[0]),list(l[1].astype(int)))
    print(l)
    cv2.line(rgb_img, l[0],l[1], (random.randint(0,255),random.randint(0,255),random.randint(0,255)),5)

def get_color(type):
    if type == "G":
        return (0,255,0)
    elif type == "R":
        return (0,0,255)
    elif type == "N":
        return (255,0,0)
    else:
        return (0,0,0)

for n in nodes:
    cv2.circle(rgb_img, (int(n[0][0]),int(n[0][1])), 10, get_color(n[1]), -1)

plt.imshow(rgb_img)
plt.show()