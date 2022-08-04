import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import cv2

from tools import detect_circles, fill_circles, turn_off_gens, get_gens_cords, get_resistors_cords, visualize_components
from flood_fill_tools import extract_largest_component

# load image and convert to grayscale
filename = "samples/small_size_kolo.png"
rgb_img = cv2.imread(filename)
gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

# negate image
negative_gray = cv2.bitwise_not(gray)

# threshold image
_, threshold_img = cv2.threshold(negative_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# extract largest component
circuit_graph_img = extract_largest_component(threshold_img)

# detect circles in image
detected_circles = detect_circles(circuit_graph_img)

# clean empty space around detected circles
cleaned_circles = fill_circles(detected_circles)

# process generators
gen_cords = get_gens_cords(cleaned_circles)

# turn off generators
circuit_wo_gens = turn_off_gens(cleaned_circles, gen_cords)

# get resistors
resistor_cords = get_resistors_cords(circuit_graph_img, circuit_wo_gens)




rgb_img = visualize_components(rgb_img, gen_cords, resistor_cords)

# display latest iteration of the algorithm
latest_img = circuit_graph_img.copy()
final_img = circuit_wo_gens.copy()
# plt.subplot(1, 2, 1)
# plt.imshow(latest_img, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(final_img, cmap='gray')
# plt.show()
plt.imshow(rgb_img)
plt.show()