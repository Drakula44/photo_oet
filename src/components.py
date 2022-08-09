from dataclasses import dataclass, field
import numpy.typing as npt
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract

import image_processing_tools as ip_tools

RADIUS = 25

@dataclass
class Name:
    symbol: str


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
    type: str = None
    symbol: str = None
    name: Name = None
    connected_notes: List = field(default_factory=list)
    isGround: bool = False
    def __eq__(self, other):
        if self.cords[0] == other.cords[0] and self.cords[1] == other.cords[1]:
            return True
        return False
    def process_name(self):
        pass

@dataclass
class Component(Node):
    pass

@dataclass
class Edge:
    edge_pair: List[Node]

@dataclass
class Generator(Component):
    name: Detail = None
    type: Detail = None
    orientation: str =  None
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
        self.name.symbol = self.type.symbol
        if data != "":
            self.name.symbol += "_" + data[0]
    def process_type(self):
        if self.type is None:
            print("No type for generator")
        if np.linalg.norm(self.type.cords-self.cords) < RADIUS:
            self.type.symbol = "I"
        else:
            self.type.symbol = "V"

@dataclass
class Resistor(Component):
    name: Detail = None
    def process_name(self):
        _ , thresh = cv2.threshold(self.name.img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        character, rest = ip_tools.split_circuit(thresh)
        rest = ip_tools.apply_mask(self.name.img, rest)
        data = pytesseract.image_to_string(rest, config='--psm 10 --oem 1 -c tessedit_char_whitelist=0123456789')
        self.name.symbol = "R"
        if data != "":
            self.name.symbol += "_" + data[0]


@dataclass
class Junction(Node):
    name: Detail = None
    potential: Detail = None
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