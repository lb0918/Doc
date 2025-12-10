import numpy as np
import pickle

liquids = ["Water","PVP50","PVP40","PVP30","PVP20","PVP10"]
temps = [0,16,21,26.5,31,37]
NIST_values_official = {}

for liquid in liquids:
    tempo = {}
    for temp in temps:
        res1 = input(f"{liquid} at {temp} degrees mean")
        res2 = input(f"{liquid} at {temp} degrees uncert")
        tempo[temp] = (res1,res2)
    NIST_values_official[liquid] = tempo
print(NIST_values_official)
with open('/home/lbsc/IRM_diffusion/NIST_values_official.pkl', 'wb') as f:
    pickle.dump(NIST_values_official, f)
