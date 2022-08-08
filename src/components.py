from dataclasses import dataclass
import numpy.typing as npt
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract

import image_processing_tools as ip_tools

RADIUS = 25


@dataclass
class Detail:
    cords: npt.ArrayLike
    bounding_box: List[npt.ArrayLike]
    img: npt.ArrayLike
    symbol: str
    detail_type: str

@dataclass
class Node:
    cords: npt.ArrayLike
    type: str
    symbol: str

    def __eq__(self, other):
        if self.cords[0] == other.cords[0] and self.cords[1] == other.cords[1]:
            return True
        return False
    def process_name(self):
        pass


@dataclass
class Edge:
    edge_pair: List[Node]

@dataclass
class Generator(Node):
    name: Detail
    type: Detail
    orientation: str
    def process_name(self):
        if self.type is None:
            print("No type for generator")
        _ , thresh = cv2.threshold(self.name.img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        character, rest = ip_tools.split_circuit(thresh)
        rest = ip_tools.apply_mask(self.name.img, rest)
        whitelist = "0123456789"
        if self.type.symbol == "I":
            whitelist += "g"
        data = pytesseract.image_to_string(rest, config='--psm 10 --oem 1 -c tessedit_char_whitelist='+whitelist)
        if self.type.symbol == "I":
            self.name.symbol = "I_" + data[0]
        else:
            self.name.symbol = "E_" + data[0]
    def process_type(self):
        if self.type is None:
            print("No type for generator")
        if np.linalg.norm(self.type.cords-self.cords) < RADIUS:
            self.type.symbol = "I"
        else:
            self.type.symbol = "E"

@dataclass
class Resistor(Node):
    name: Detail
    def process_name(self):
        _ , thresh = cv2.threshold(self.name.img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        character, rest = ip_tools.split_circuit(thresh)
        rest = ip_tools.apply_mask(self.name.img, rest)
        data = pytesseract.image_to_string(rest, config='--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789')
        self.name.symbol = "R_" + data[0]


@dataclass
class Junction(Node):
    name: Detail
    potential: Detail
    def process_name(self):
        data = pytesseract.image_to_string(self.name.img, config='--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789')
        self.name.symbol = data[0]

    def process_potential(self):
        if self.potential is None:
            print("No potential for junction")
        _ , thresh = cv2.threshold(self.potential.img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        character, rest = ip_tools.split_circuit(thresh)
        rest = ip_tools.apply_mask(self.potential.img, rest)
        data = pytesseract.image_to_string(rest, config='--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789')
        self.potential.symbol = "V_"+data[0]