from pymoab import core, topo_util
from geometric import get_tetra_volume
from math import sin, pi
import numpy as np

mb = core.Core()
mtu = topo_util.MeshTopoUtil(mb)
mb.load_file('src/test_case_2_eymard_result.h5m')
volumes = mb.get_entities_by_dimension(0, 3)
print("# of volumes: {}".format(len(volumes)))

pressure_tag = mb.tag_get_handle('PRESSURE')
centroid_tag = mb.tag_get_handle('CENTROID')
est_u = mb.tag_get_data(pressure_tag, volumes).reshape(len(volumes))
centroids = mb.tag_get_data(centroid_tag, volumes)

u_solution = lambda x, y, z: x + y + z
exact_u = np.array([u_solution(c[0], c[1], c[2]) for c in centroids])

l2_denom, l2_num = [], []
for u1, u2, vol in zip(est_u, exact_u, volumes):
    tetra_nodes = mtu.get_bridge_adjacencies(vol, 3, 0)
    tetra_nodes = mb.get_coords(tetra_nodes).reshape(4, 3)
    tetra_vol = get_tetra_volume(tetra_nodes)
    l2_num.append(((u2 - u1)**2)*tetra_vol)
    l2_denom.append((u2**2)*tetra_vol)

l2_norm = (sum(l2_num) / sum(l2_denom))**0.5
print("l2-norm = {}".format(l2_norm))
