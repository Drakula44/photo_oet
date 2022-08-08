import numpy as np
import matplotlib.pyplot as plt
import cv2
import random  

from components import *
from image_processing import process_image
from image_processing_tools import display_changes
from tools import *



filename = "samples/small_size_kolo.png"
rgb_img = cv2.imread(filename)
gen_cords, resistor_cords, nodes, lines, details_cords = process_image(rgb_img.copy())

gens = [Generator(gen[0],None, None, None, None) for gen in gen_cords]
resistors = [Resistor(res[0], None, None, None) for res in resistor_cords]
junctions = [Junction(node, None, None, None, None) for node in nodes]
details = [Detail(cords=(detail[0]+detail[1])//2,bounding_box=[detail[0],detail[1]],img=detail[-1],symbol=None,detail_type=None) for detail in details_cords]

del gen_cords
del resistor_cords
del details_cords
del nodes

lines = split_lines(lines, junctions)
lines = merge_point(lines, junctions)

unique_nodes = np.unique(np.array([l[0] for l in lines] + [l[1] for l in lines]),axis=0)
nodes = [Node(node,None, None) for node in unique_nodes if Node(node,None, None) not in junctions]
nodes += junctions

lines, nodes = add_components(lines, nodes, resistors)
lines, nodes = add_components(lines, nodes, gens)

assign_details(nodes, details, rgb_img)

plt.imshow(visualize_graph(lines, nodes, rgb_img))
plt.show()