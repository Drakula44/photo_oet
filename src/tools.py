from re import L
from statistics import correlation
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random  

from components import *
from image_processing_tools import display_changes

# radius of generator should not be constant but for now
RADIUS = 25

# refactored
def get_color(node):
    if isinstance(node, Generator):
        if node.type.symbol == "I":
            return (255,255,0)
        else:
            return (0,255,255)
    elif isinstance(node, Resistor):
        return (0,0,255)
    elif isinstance(node, Junction):
        return (255,0,0)
    else:
        return (0,0,0)

# refactored?
def visualize_graph(lines: List[npt.ArrayLike], nodes: List[Node], img: npt.ArrayLike, visual_text= False):
    img = img.copy()
    for line in lines:
        line = (list(line[0].astype(int)),list(line[1].astype(int)))
        cv2.line(img, line[0],line[1], (random.randint(0,255),random.randint(0,255),random.randint(0,255)),5)
    for node in nodes:
        
        cv2.circle(img, node.cords, 10, get_color(node), -1)
        if not visual_text:
            continue
        if node.name is None:
            continue
        cv2.putText(img, node.name.symbol, node.cords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return img

# refactored
def close_enoug_to_line(line: npt.ArrayLike, node: Node):
    node_cords = node.cords
    d1 = np.linalg.norm(line[0]-node_cords)
    d2 = np.linalg.norm(line[1]-node_cords)
    r = np.linalg.norm(line[1]-line[0])
    d= np.linalg.norm(np.cross(line[0]-line[1], node_cords-line[1]))/np.linalg.norm(line[0]-line[1])
    return d < 3 and d1/r < 0.8 and d2/r < 0.8

def merge_point(lines, junctions):
    for l1 in range(len(lines)):
        for l2 in range(len(lines)):
            if close_enough(lines[l1][0], lines[l2][0]):
                if Node(lines[l1][0]) in junctions:
                    lines[l2][0] = lines[l1][0]
                else:
                    lines[l1][0] = lines[l2][0]
            if close_enough(lines[l1][0], lines[l2][1]):
                if Node(lines[l1][0]) in junctions:
                    lines[l2][1] = lines[l1][0]
                else:
                    lines[l1][0] = lines[l2][1]
            if close_enough(lines[l1][1], lines[l2][0]):
                if Node(lines[l1][1]) in junctions:
                    lines[l2][0] = lines[l1][1]
                else:
                    lines[l1][1] = lines[l2][0]
            if close_enough(lines[l1][1], lines[l2][1]):
                if Node(lines[l1][1]) in junctions:
                    lines[l2][1] = lines[l1][1]
                else:
                    lines[l1][1] = lines[l2][1]
    return lines


def split_lines(lines :List[npt.ArrayLike], nodes: List[Node]):
    lines_split = []
    for line in lines:
        one_split = False
        for node in nodes:
            if close_enoug_to_line(line, node):
                one_split = True
                lines_split.append([line[0],node.cords])
                lines_split.append([node.cords,line[1]])
        if not one_split:
            lines_split.append(line)
    return lines_split

def close_enough(p1, p2):
    return np.linalg.norm(p1-p2) < 5

# FUTURE
# if component have more then 2 connection points this wont work
def add_components(lines, nodes: List[Node], components: List[Node]):
    splited_lines = []
    for line in lines:
        one_split = False
        for component in components:
            if close_enoug_to_line(line, component):
                one_split = True
                splited_lines.append([line[0],component.cords])
                splited_lines.append([component.cords,line[1]])
                nodes.append(component)
        if not one_split:
            splited_lines.append(line)
    return splited_lines, nodes

def find_closest_detail(node, details):
    closest_detail = None
    closest_dist = None
    for i, detail in enumerate(details):
        detail = detail.cords
        dist = np.linalg.norm(node-detail)
        if closest_dist is None or dist < closest_dist:
            closest_dist = dist
            closest_detail = i
    return closest_detail

def find_closest_node(detail, nodes):
    closest_node = None
    closest_dist = None
    for i, node in enumerate(nodes):
        node = node.cords
        dist = np.linalg.norm(detail-node)
        if closest_dist is None or dist < closest_dist:
            closest_dist = dist
            closest_node = i
    return closest_node

def assign_details(nodes, details, rgb_img):
    for node in nodes:
        if not isinstance(node, Resistor):
            continue
        index_closest = find_closest_detail(node.cords, details)
        node.name=details[index_closest]
        node.process_name()
        details.pop(index_closest)

    for node in nodes:
        if not isinstance(node, Generator):
            continue
        # asuming that + sign is closer to the gen then name
        index_closest = find_closest_detail(node.cords, details)
        node.type = details[index_closest]
        node.process_type()
        details.pop(index_closest)

        index_closest = find_closest_detail(node.cords, details)
        node.name = details[index_closest]
        node.process_name()
        details.pop(index_closest)

    for node in nodes:
        if not isinstance(node, Junction):
            continue
        index_closest = find_closest_detail(node.cords, details)
        if index_closest is None:
            break
        node.name = details[index_closest]
        node.process_name()
        details.pop(index_closest)

    for detail in details:
        index_node = find_closest_node(detail.cords, nodes)
        if index_node is None:
            break
        if isinstance(nodes[index_node], Junction):
            nodes[index_node].potential = detail
            nodes[index_node].process_potential()
        else:
            print("Not processed")
    
        
    return nodes


def add_nodes_between_components(lines, nodes):
    for_poping = []
    for i, line in enumerate(lines):
        id1 = nodes.index(Node(line[0]))
        id2 = nodes.index(Node(line[1]))
        if isinstance(nodes[id1], Component) and isinstance(nodes[id2], Component):
            new_node = ((line[0]+line[1])/2).astype(int)
            nodes.append(Node(new_node))
            lines.append([line[0],new_node])
            lines.append([new_node,line[1]])
            for_poping.append(i)
    for_poping.reverse()
    for i in for_poping:
        lines.pop(i)

def add_nodes_names(nodes):
    for node in nodes:
        if node.name is not None:
            continue
        node.name = Name("_" + str(node.cords[0])+"_"+str(node.cords[1]))

def get_orientation(node):
    if node.type.symbol == "I":
        fliph = cv2.flip(node.type.img, 1)
        flipv = cv2.flip(node.type.img, -1)
        correlation_h = np.abs(np.abs(node.type.img - fliph))
        correlation_v = np.abs(np.abs(node.type.img - flipv))
        if correlation_h.sum() < correlation_v.sum():
            hist = np.sum(node.type.img, axis=1)
            # get sum of first half and second half of the hist
            sum_first = np.sum(hist[:int(len(hist)/2)])
            sum_second = np.sum(hist[int(len(hist)/2):])
            if sum_first > sum_second and np.sum(node.connected_notes[0].cords) > np.sum(node.connected_notes[1].cords):
                node.connected_notes.reverse()
            elif sum_first < sum_second and np.sum(node.connected_notes[0].cords) < np.sum(node.connected_notes[1].cords):
                node.connected_notes.reverse()
        else:
            hist = np.sum(node.type.img, axis=0)
            # get sum of first half and second half of the hist
            sum_first = np.sum(hist[:int(len(hist)/2)])
            sum_second = np.sum(hist[int(len(hist)/2):])
            if sum_first > sum_second and np.sum(node.connected_notes[0].cords) > np.sum(node.connected_notes[1].cords):
                node.connected_notes.reverse()
            elif sum_first < sum_second and np.sum(node.connected_notes[0].cords) < np.sum(node.connected_notes[1].cords):
                node.connected_notes.reverse()
    else:
        d1 = np.linalg.norm(node.type.cords - node.connected_notes[0].cords)
        d2 = np.linalg.norm(node.type.cords - node.connected_notes[0].cords)
        vector_1 = node.cords - node.type.cords
        vector_2 = node.connected_notes[0].cords - node.type.cords
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle1 = np.arccos(dot_product)
        vector_1 = node.cords - node.type.cords
        vector_2 = node.connected_notes[1].cords - node.type.cords
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle2 = np.arccos(dot_product)
        if angle1 < angle2:
            node.connected_notes.reverse()
    return node

def create_graph_txt(lines, nodes):
    text = ""
    for i, line in enumerate(lines):
        id1 = nodes.index(Node(line[0]))
        id2 = nodes.index(Node(line[1]))
        if isinstance(nodes[id1], Component): 
            nodes[id1].connected_notes.append(nodes[id2])
        if isinstance(nodes[id2], Component):
            nodes[id2].connected_notes.append(nodes[id1])
    
    # create text for components
    for node in nodes:
        if not isinstance(node, Component):
            continue
        text += node.name.symbol + " "
        if len(node.connected_notes) != 2:
            print("Not connected to 2 nodes", node.name.symbol)
        if isinstance(node, Generator):
            get_orientation(node)
        for connected_node in node.connected_notes:
            text += connected_node.name.symbol + " "
        text += ";"

            
        delta_cord = node.connected_notes[1].cords - node.connected_notes[0].cords
        if np.argmax(np.abs(delta_cord)) == 0:
            if delta_cord[0] > 0:
                text += " right"
            else:
                text += " left"
        else:
            if delta_cord[1] > 0:
                text += " down"
            else:
                text += " up"
        text += "\n" 
    for line in lines:
        id1 = nodes.index(Node(line[0]))
        id2 = nodes.index(Node(line[1]))
        if isinstance(nodes[id1], Component) or isinstance(nodes[id2], Component): 
            continue        
        text += "W "
        text += nodes[id1].name.symbol + " "
        text += nodes[id2].name.symbol + " ;"
        delta_cord = nodes[id2].cords - nodes[id1].cords
        if np.argmax(np.abs(delta_cord)) == 0:
            if delta_cord[0] > 0:
                text += " right"
            else:
                text += " left"
        else:
            if delta_cord[1] > 0:
                text += " down"
            else:
                text += " up"
        if nodes[id1].isGround or nodes[id2].isGround:
            text += ", sground"
        text += "\n"
    print(text)
    return text