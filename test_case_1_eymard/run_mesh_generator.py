from pymoab import core, rng, types
from mesh_generator import MeshGenerator
from sympy import sin, pi, lambdify, Matrix
from sympy.vector import CoordSys3D, Del
from math import atan
import numpy as np
from numpy.linalg import inv

C = CoordSys3D('C')
nabla = Del()
u_field = 1 + sin(pi*C.x)*sin(pi*(C.y + 0.5))*sin(pi*(C.z + 1/3))
grad_u = nabla(u_field, doit=True)
u_func = lambdify([C.x, C.y, C.z], u_field)

K = np.array([1.0, 0.5, 0.0,
              0.5, 1.0, 0.5,
              0.0, 0.5, 1.0]).reshape(3, 3)

print("Generating mesh...")
mesh_gen = MeshGenerator('../geometrical_models/tet_8.h5m')
vertices = mesh_gen.mb.get_entities_by_dimension(0, 0)
mesh_gen.mtu.construct_aentities(vertices)
mesh_gen.set_centroids()
mesh_gen.set_homogeneous_permeability(K.reshape(9))

grad_u_matrix = Matrix([grad_u.coeff(C.i), grad_u.coeff(C.j), grad_u.coeff(C.k)])
F = Matrix(K) * grad_u_matrix
div_F = F.diff(C.x)[0] + F.diff(C.y)[1] + F.diff(C.z)[2]
div_F_func = lambdify([C.x, C.y, C.z], div_F)
mesh_gen.set_source_term(div_F_func)

mesh_gen.set_dirichlet_boundary_conditions(u_func)

print("Done")

mesh_gen.mb.write_file('test_case_1_eymard.h5m')
