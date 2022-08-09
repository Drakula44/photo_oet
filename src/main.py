import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from lcapy import Circuit


from components import *
from image_processing import process_image
from image_processing_tools import display_changes
from tools import *



filename = "samples/small_size_kolo.png"
rgb_img = cv2.imread(filename)
gen_cords, resistor_cords, nodes, lines, details_cords = process_image(rgb_img.copy())

# ovo treba uraditi cim se prvi put izvdovje komponente
gens = [Generator(gen[0]) for gen in gen_cords]
resistors = [Resistor(res[0]) for res in resistor_cords]
junctions = [Junction(node) for node in nodes]
details = [Detail(cords=(detail[0]+detail[1])//2,bounding_box=[detail[0],detail[1]],img=detail[-1]) for detail in details_cords]

del gen_cords
del resistor_cords
del details_cords
del nodes

# junction mi je bilo koji cvor u kolu
# i ovo split je da se linije podeli na dva dela ako sadrzi junction u sredini
lines = split_lines(lines, junctions)

# sve nodove koji su relativno blizu treba staviit da imaju jedne komentare
lines = merge_point(lines, junctions)

# izdvojiti sve uniqe nodove 
# ovo bi trebalo da da potpun graf kola
unique_nodes = np.unique(np.array([l[0] for l in lines] + [l[1] for l in lines]),axis=0)
nodes = [Node(node) for node in unique_nodes if Node(node) not in junctions]
nodes += junctions

# sad dodati komponente
lines, nodes = add_components(lines, nodes, resistors)
lines, nodes = add_components(lines, nodes, gens)

# svakoj komponenti dodeliti detalje koji su jos najbiliz
# najverovatnije jedan od funckija sa najvise dorada kako ima mnogo special caseva
# npr. u ovo slici sada potencijal V5 isto tako ovde su svi cvorovi oznaceni dok u nekima nisu i onda to moze da zabode program itd
assign_details(nodes, details, rgb_img)

# za redne veze komponenti treba dodati node izmedju
# zbog nacina na koji se kasnije crta graph
add_nodes_between_components(lines, nodes)

# sve nodovi koji nisu otpornici, generatori ili cvoorvi nemaju ime pa im dam ime kao prilepljene kordinate
# _ pre imena i nece se prikazivati na slici kao oznaka noda
add_nodes_names(nodes)
# plt.imshow(visualize_graph(lines, nodes, rgb_img))
# plt.show()

# ok ovo je sad hard codovao da je drugi cvor uzemljenje 
# inace kao cvor za uzemljenje treba da ima naziv nula 
# e sad to treba videti da li moze da se promeni posto sam ja nesto gledao ali nisam nasao
for node in nodes:
    if node.name.symbol == "2":
        node.name.symbol = "0"

# samo pravim text koji posle prosledim funkciji
text = create_graph_txt(lines, nodes)

circuit = Circuit(text)
circuit.draw("test.pdf")

# ok sigurno postoji laksi nacin kako postaviti vrednost otoprnika i slicno
#  jer ovako mu treba tri godine da nadje jednacine jer sve resava simbolicki a za to nema potrebe
eq = circuit.I_g.v
print(eq)
eq = eq.subs("V_1", 12)
eq = eq.subs("V_2", 42)
eq = eq.subs("V_4", 18)
eq = eq.subs("I_g", 60*10**-3)
eq = eq.subs("R_2", 200)
eq = eq.subs("R_3", 280)
eq = eq.subs("R_5", 300)
eq = eq.subs("R_6", 1000)
eq = eq.subs("R_7", 200)
print(eq)