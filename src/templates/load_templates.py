import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from templates.template import Template

def prepare_template(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_img = gray
    gray = cv2.bitwise_not(gray)
    kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=1)
    return template_img

def prepare_mask(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bit_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bit_mask = bit_mask.astype(np.uint8)
    return bit_mask

def get_connections_from_file(filename):
    with open(filename, "r") as f:
        connections = f.read().splitlines()
        connections = [[[int(k) for k in j.split("-")] for j in i.split(" ")] for i in connections]
    return connections

def read_template(filename):
    template_img = prepare_template(filename)
    # invert tempalte image
    template_img = cv2.bitwise_not(template_img)
    connections_filename = filename.replace(".png", ".txt")
    connections = get_connections_from_file(connections_filename)
    mask_filename = filename.replace(".txt", "_mask.png")
    mask = prepare_mask(mask_filename)
    name = filename.split("/")[-1].split(".")[0]
    return Template(template_img, mask, name, connections)

def load_templates(folder_name):
    templates = []
    template_filenames = os.listdir(folder_name)
    # filter only png files
    template_filenames = [i for i in template_filenames if i.endswith(".png")]
    # filter out names that have mask in it
    template_filenames = [i for i in template_filenames if "mask" not in i]
    template_filenames.sort()
    for template_filename in template_filenames:
        templates.append(read_template(folder_name + "/" + template_filename))
    return templates
    