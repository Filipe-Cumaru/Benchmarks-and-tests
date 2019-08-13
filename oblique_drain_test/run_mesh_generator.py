from pymoab import core, rng, types
from mesh_generator import MeshGenerator
from sympy import sin, pi, lambdify, Matrix
from sympy.vector import CoordSys3D, Del
from math import atan, isclose
import numpy as np
from numpy.linalg import inv

C = CoordSys3D('C')
nabla = Del()
u_field = -C.x - 0.2*C.y
grad_u = nabla(u_field, doit=True)
u_func = lambdify([C.x, C.y, C.z], u_field)

M1 = np.array([100.0, 0.0, 0.0, \
                0.0, 10.0, 0.0, \
                0.0, 0.0, 1.0]).reshape(3, 3)
M2 = np.array([1.0, 0.0, 0.0, \
                0.0, 0.1, 0.0, \
                0.0, 0.0, 1.0]).reshape(3, 3)
theta = atan(0.2)
rot_z = np.array([np.cos(theta), np.sin(theta), 0.0, \
                -np.sin(theta), np.cos(theta), 0.0, \
                0.0, 0.0, 1.0]).reshape(3, 3)
rot_z_inv = inv(rot_z)
K1 = ((rot_z*M1)*rot_z_inv)
K2 = ((rot_z*M2)*rot_z_inv)

print("Generating mesh...")
mesh_gen = MeshGenerator('oblique_drain.h5m')
vertices = mesh_gen.mb.get_entities_by_dimension(0, 0)
mesh_gen.mtu.construct_aentities(vertices)
mesh_gen.set_centroids()

phi_1 = lambda x, y: y - 0.2*(x - 0.5) - 0.475
phi_2 = lambda x, y: phi_1(x, y) - 0.05
omega_1, omega_2, omega_3 = [], [], []
volumes = mesh_gen.mb.get_entities_by_dimension(0, 3)
centroids = mesh_gen.mb.tag_get_data(mesh_gen.centroid_tag, volumes)
for vol, c in zip(volumes, centroids):
    if phi_1(c[0], c[1]) < 0:
        omega_1.append(vol)
    elif phi_1(c[0], c[1]) > 0 and phi_2(c[0], c[1]) < 0:
        omega_2.append(vol)
    elif phi_1(c[0], c[1]) > 0:
        omega_3.append(vol)
mesh_gen.set_homogeneous_permeability(K1.reshape(9), vols=omega_2)
mesh_gen.set_homogeneous_permeability(K2.reshape(9), vols=omega_1)
mesh_gen.set_homogeneous_permeability(K2.reshape(9), vols=omega_3)

grad_u_matrix = Matrix([grad_u.coeff(C.i), grad_u.coeff(C.j), grad_u.coeff(C.k)])

F1 = Matrix(K1) * grad_u_matrix
div_F1 = F1.diff(C.x)[0] + F1.diff(C.y)[1] + F1.diff(C.z)[2]
div_F1_func = lambdify([C.x, C.y, C.z], div_F1)
mesh_gen.set_source_term(div_F1_func, vols=omega_2)

F2 = Matrix(K2) * grad_u_matrix
div_F2 = F2.diff(C.x)[0] + F2.diff(C.y)[1] + F2.diff(C.z)[2]
div_F2_func = lambdify([C.x, C.y, C.z], div_F2)
mesh_gen.set_source_term(div_F2_func, vols=(omega_1 + omega_3))

dirichlet_faces = []
neumann_faces = []
all_faces = mesh_gen.mb.get_entities_by_dimension(0, 2)
for face in all_faces:
    adj_vols = mesh_gen.mtu.get_bridge_adjacencies(face, 2, 3)
    I, J, K = mesh_gen.mtu.get_bridge_adjacencies(face, 2, 0)
    JI = mesh_gen.mb.get_coords([I]) - mesh_gen.mb.get_coords([J])
    JK = mesh_gen.mb.get_coords([K]) - mesh_gen.mb.get_coords([J])
    normal_area_vec = np.cross(JI, JK)
    if len(adj_vols) < 2:
        dot_product = np.dot(normal_area_vec, np.array([0.0, 0.0, 1.0]))
        if dot_product == 0.0:
            dirichlet_faces.append(face)
        else:
            if mesh_gen.mb.get_coords([K])[2] == 0.0 or \
                mesh_gen.mb.get_coords([K])[2] == 1.0:
                neumann_faces.append(face)
            else:
                dirichlet_faces.append(face)
mesh_gen.set_dirichlet_boundary_conditions(u_func, dirichlet_faces=dirichlet_faces)
mesh_gen.set_neumann_boundary_conditions(grad_u_matrix, neumann_faces=neumann_faces)

print("Done")

dirichlet_set = mesh_gen.mb.create_meshset()
mesh_gen.mb.add_entities(dirichlet_set, dirichlet_faces)
mesh_gen.mb.write_file('test_oblique_drain_dirichlet.h5m', output_sets=(dirichlet_set,))

neumann_set = mesh_gen.mb.create_meshset()
mesh_gen.mb.add_entities(neumann_set, neumann_faces)
mesh_gen.mb.write_file('test_oblique_drain_neumann.h5m', output_sets=(neumann_set,))

# mesh_gen.mb.write_file('test_oblique_drain_coarse.h5m')
mesh_gen.mb.write_file('test_oblique_drain.h5m')
