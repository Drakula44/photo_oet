import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import cv2
import pytesseract

from tools import *
from flood_fill_tools import *

def process_image(rgb_img):
    # load image and convert to grayscale
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    # negate image
    negative_gray = cv2.bitwise_not(gray)

    # threshold image
    _, threshold_img = cv2.threshold(negative_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # extract largest component
    circuit_graph_img, details_img = split_circuit(threshold_img)
    gray_details = (details_img.astype(float) * negative_gray.astype(float) / 255).astype(np.uint8)

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

    # turn off components
    circuit_wo_components = turn_off_components(circuit_graph_img.copy(), resistor_cords, gen_cords)

    # find lines in image
    lines = find_lines(circuit_wo_components)

    # extract junctions
    nodes = extract_junctions(circuit_graph_img.copy())

    # create clusters of symbols
    clusterd_details = cluster_symbols(gray_details)

    # find bounding boxes of symbols
    details = find_bounding_box_details(clusterd_details)

    # structure elements properlly
    gen_cords = [[np.array([i[0],i[1]]),"G"] for i in gen_cords]
    resistor_cords = [[np.array(i[0]),"R"] for i in resistor_cords]
    nodes = [np.array(i) for i in nodes]
    lines = [[np.array([i[0][0],i[0][1]]), np.array([i[0][2],i[0][3]]), "N"] for i in lines]

    # assign components to details
    # gen_cords, resistor_cords, nodes = assign_details(details, gen_cords, resistor_cords, nodes)


    plt.imshow(rgb_img)
    plt.show()

    # display latest iteration of the algorithm
    latest_img = rgb_img.copy()
    final_img = circuit_wo_gens.copy()
    plt.subplot(1, 2, 1)
    plt.imshow(details_img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(gray_details, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(negative_gray, cmap='gray')
    plt.show()

    return gen_cords, resistor_cords, nodes, lines


if __name__ == "__main__":

    filename = "samples/small_size_kolo.png"
    rgb_img = cv2.imread(filename)
    process_image(rgb_img)

