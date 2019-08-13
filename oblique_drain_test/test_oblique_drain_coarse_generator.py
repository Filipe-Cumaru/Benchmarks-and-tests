import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util


mb = core.Core()
mtu = topo_util.MeshTopoUtil(mb)

material_set_tag = mb.tag_get_handle(
    "MATERIAL_SET", 1, types.MB_TYPE_INTEGER,
    types.MB_TAG_SPARSE, True)

coords = np.array([[0.000, 0.000, 0.000],
                   [1.000, 0.000, 0.000],
                   [0.000, 0.375, 0.000],
                   [1.000, 0.575, 0.000],
                   [0.000, 0.000, 1.000],
                   [1.000, 0.000, 1.000],
                   [0.000, 0.375, 1.000],
                   [1.000, 0.575, 1.000],
                   [1.000, 1.000, 0.000],
                   [1.000, 0.580, 0.000],
                   [0.000, 0.380, 0.000],
                   [0.000, 1.000, 0.000],
                   [0.000, 0.380, 1.000],
                   [1.000, 0.580, 1.000],
                   [0.000, 1.000, 1.000],
                   [1.000, 1.000, 1.000]
                   ])

verts = mb.create_vertices(coords.flatten())

# The Lower formation
tetra1 = mb.create_element(types.MBTET, [verts[0], verts[1],
                                         verts[3], verts[5]])
tetra2 = mb.create_element(types.MBTET, [verts[0], verts[4],
                                         verts[5], verts[6]])
tetra3 = mb.create_element(types.MBTET, [verts[0], verts[2],
                                         verts[3], verts[6]])
tetra4 = mb.create_element(types.MBTET, [verts[5], verts[6],
                                         verts[3], verts[7]])
tetra5 = mb.create_element(types.MBTET, [verts[0], verts[6],
                                         verts[3], verts[5]])

# The drain tetrahedra
tetra6 = mb.create_element(types.MBTET, [verts[6], verts[7],
                                         verts[3], verts[13]])
tetra7 = mb.create_element(types.MBTET, [verts[2], verts[3],
                                         verts[6], verts[10]])
tetra8 = mb.create_element(types.MBTET, [verts[6], verts[12],
                                         verts[10], verts[13]])
tetra9 = mb.create_element(types.MBTET, [verts[9], verts[3],
                                         verts[13], verts[10]])
tetra10 = mb.create_element(types.MBTET, [verts[13], verts[10],
                                          verts[3], verts[6]])

# The Upper formation
tetra11 = mb.create_element(types.MBTET, [verts[10], verts[8],
                                          verts[9], verts[13]])
tetra12 = mb.create_element(types.MBTET, [verts[10], verts[14],
                                          verts[13], verts[12]])
tetra13 = mb.create_element(types.MBTET, [verts[11], verts[14],
                                          verts[8], verts[10]])
tetra14 = mb.create_element(types.MBTET, [verts[15], verts[8],
                                          verts[14], verts[13]])
tetra15 = mb.create_element(types.MBTET, [verts[14], verts[10],
                                          verts[13], verts[8]])
all_verts = mb.get_entities_by_dimension(0, 0)
mtu.construct_aentities(all_verts)

mb.write_file('oblique_drain_coarse.h5m')
