from lcapy import Circuit
import matplotlib.pyplot as plt

cct = Circuit("""R_2 2 _182_126 ; up
R_7 _293_330 3 ; up
R_3 1 _58_35 ; up
R_5 1 2 ; right
R_6 2 3 ; right
I_g 5 2 ; up
V_2 4 _182_126 ; down
V_4 3 _293_35 ; up
V_1 _58_330 1 ; up
W _58_330 5 ; right
W 5 _293_330 ; right
W _58_35 4 ; right
W 4 _293_35 ; right
""")
cct.draw("test.pdf")