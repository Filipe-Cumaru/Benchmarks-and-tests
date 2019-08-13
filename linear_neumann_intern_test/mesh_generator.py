import numpy as np
from numpy.linalg import norm
from pymoab import core, rng
from pymoab import types
from pymoab import topo_util
from sympy.vector import CoordSys3D
from geometric import get_tetra_volume


class MeshGenerator:

    def __init__(self, mesh_file):
        self.mesh_file = mesh_file
        self.mb = core.Core()
        self.mb.load_file(mesh_file)
        self.mtu = topo_util.MeshTopoUtil(self.mb)
        self.dirichlet_tag = self.mb.tag_get_handle(
            "DIRICHLET", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)
        self.neumann_tag = self.mb.tag_get_handle(
            "NEUMANN", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)
        self.permeability_tag = self.mb.tag_get_handle(
            "PERMEABILITY", 9, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)
        self.source_tag = self.mb.tag_get_handle(
            "SOURCE", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)
        self.centroid_tag = self.mb.tag_get_handle(
            "CENTROID", 3, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

    def set_homogeneous_permeability(self, K, vols=None):
        if vols is None:
            vols = self.mb.get_entities_by_dimension(0, 3)
        permeability = np.tile(K, len(vols))
        self.mb.tag_set_data(self.permeability_tag, vols, permeability)

    def set_heterogeneous_permeability(self, k_func, vols=None):
        if vols is None:
            vols = self.mb.get_entities_by_dimension(0, 3)
        centroids = self.mb.tag_get_data(self.centroid_tag, vols)
        permeability = [k_func(c[0], c[1], c[2]) for c in centroids]
        self.mb.tag_set_data(self.permeability_tag, vols, permeability)

    def set_centroids(self):
        all_volumes = self.mb.get_entities_by_dimension(0, 3)
        centroids = [self.mtu.get_average_position([vol]) for vol in all_volumes]
        self.mb.tag_set_data(self.centroid_tag, all_volumes, centroids)

    def set_source_term(self, q_func, vols=None):
        if vols is None:
            vols = self.mb.get_entities_by_dimension(0, 3)
        source_terms = []
        for vol in vols:
            center = self.mb.tag_get_data(self.centroid_tag, vol)[0]
            vol_nodes = self.mtu.get_bridge_adjacencies(vol, 3, 0)
            vol_nodes = self.mb.get_coords(vol_nodes).reshape(4, 3)
            volume = get_tetra_volume(vol_nodes)
            source_terms.append(volume*q_func(center[0], center[1], center[2]))
        self.mb.tag_set_data(self.source_tag, vols, source_terms)

    def set_dirichlet_boundary_conditions(self, bc_func, dirichlet_faces=None):
        if dirichlet_faces is None:
            all_faces = self.mb.get_entities_by_dimension(0, 2)
            dirichlet_faces = [face for face in all_faces \
                    if len(self.mtu.get_bridge_adjacencies(face, 2, 3)) < 2]
        self.mb.tag_set_data(self.dirichlet_tag, dirichlet_faces, np.zeros(len(dirichlet_faces)))
        dirichlet_nodes = self.mtu.get_bridge_adjacencies(rng.Range(dirichlet_faces), 2, 0)
        dirichlet_nodes_coords = self.mb.get_coords(dirichlet_nodes).reshape(len(dirichlet_nodes), 3)
        bc_values = [bc_func(node[0], node[1], node[2]) for node in dirichlet_nodes_coords]
        self.mb.tag_set_data(self.dirichlet_tag, dirichlet_nodes, bc_values)

    def set_neumann_boundary_conditions(self, grad_u, neumann_faces=None):
        C = CoordSys3D('C')
        if neumann_faces is None:
            all_faces = self.mb.get_entities_by_dimension(0, 2)
            neumann_faces = [face for face in all_faces \
                    if len(self.mtu.get_bridge_adjacencies(face, 2, 3)) < 2]
        bc_values = []
        for face in neumann_faces:
            I, J, K = self.mtu.get_bridge_adjacencies(face, 2, 0)
            avg_pos = self.mtu.get_average_position([I, J, K])
            JI = self.mb.get_coords([I]) - self.mb.get_coords([J])
            JK = self.mb.get_coords([K]) - self.mb.get_coords([J])
            normal_area_vec = np.cross(JI, JK)
            normal_area_vec /= norm(normal_area_vec)
            grad_u_face = np.array(grad_u.subs({C.x: avg_pos[0], \
                                                C.y: avg_pos[1], \
                                                C.z: avg_pos[2]}), dtype=float).reshape(3)
            bc_values.append(np.dot(grad_u_face, normal_area_vec))
        self.mb.tag_set_data(self.neumann_tag, neumann_faces, bc_values)
        neumann_nodes = self.mtu.get_bridge_adjacencies(rng.Range(neumann_faces), 2, 0)
        self.mb.tag_set_data(self.neumann_tag, neumann_nodes, np.zeros(len(neumann_nodes)))
