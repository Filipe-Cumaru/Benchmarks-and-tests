from pymoab import core, rng, types
from mesh_generator import MeshGenerator
from sympy import sin, pi, lambdify, Matrix
from sympy.vector import CoordSys3D, Del
from math import atan
import numpy as np
from numpy.linalg import inv

C = CoordSys3D('C')
nabla = Del()
u_field = -C.x
grad_u = nabla(u_field, doit=True)
u_func = lambdify([C.x, C.y, C.z], u_field)

K = np.array([1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0]).reshape(3, 3)

print("Generating mesh...")
mesh_gen = MeshGenerator('benchmark_test_case_5.h5m')
vertices = mesh_gen.mb.get_entities_by_dimension(0, 0)
mesh_gen.mtu.construct_aentities(vertices)
mesh_gen.set_centroids()
mesh_gen.set_homogeneous_permeability(K.reshape(9))

grad_u_matrix = Matrix([grad_u.coeff(C.i), grad_u.coeff(C.j), grad_u.coeff(C.k)])
F = Matrix(K) * grad_u_matrix
div_F = F.diff(C.x)[0] + F.diff(C.y)[1] + F.diff(C.z)[2]
div_F_func = lambdify([C.x, C.y, C.z], div_F)
mesh_gen.set_source_term(div_F_func)

dirichlet_left_faces = []
dirichlet_right_faces = []
neumann_faces = []
all_faces = mesh_gen.mb.get_entities_by_dimension(0, 2)
for face in all_faces:
    adj_vols = mesh_gen.mtu.get_bridge_adjacencies(face, 2, 3)
    I, J, K = mesh_gen.mtu.get_bridge_adjacencies(face, 2, 0)
    JI = mesh_gen.mb.get_coords([I]) - mesh_gen.mb.get_coords([J])
    JK = mesh_gen.mb.get_coords([K]) - mesh_gen.mb.get_coords([J])
    normal_area_vec = np.cross(JI, JK)
    if len(adj_vols) < 2:
        if np.dot(normal_area_vec, np.array([1.0, 0.0, 0.0])) == 0.0:
            neumann_faces.append(face)
        else:
            if mesh_gen.mb.get_coords([I])[0] == 0.0:
                dirichlet_left_faces.append(face)
            elif mesh_gen.mb.get_coords([I])[0] == 1.0:
                dirichlet_right_faces.append(face)
mesh_gen.set_dirichlet_boundary_conditions(u_func, dirichlet_faces=dirichlet_left_faces)
mesh_gen.set_dirichlet_boundary_conditions(u_func, dirichlet_faces=dirichlet_right_faces)
mesh_gen.set_neumann_boundary_conditions(grad_u_matrix, neumann_faces=neumann_faces)

print("Done")

dirichlet_set = mesh_gen.mb.create_meshset()
mesh_gen.mb.add_entities(dirichlet_set, (dirichlet_left_faces + dirichlet_right_faces))
mesh_gen.mb.write_file('dirichlet_faces.h5m', output_sets=(dirichlet_set,))

neumann_set = mesh_gen.mb.create_meshset()
mesh_gen.mb.add_entities(neumann_set, neumann_faces)
mesh_gen.mb.write_file('neumann_faces.h5m', output_sets=(neumann_set,))

mesh_gen.mb.write_file('test_linear_intern_neumann.h5m')
