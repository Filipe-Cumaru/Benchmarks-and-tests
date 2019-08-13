from pymoab import core, rng, types
from mesh_generator import MeshGenerator
from sympy import sin, pi, lambdify, Matrix
from sympy.vector import CoordSys3D, Del
from math import atan
import numpy as np
from numpy.linalg import inv

C = CoordSys3D('C')
nabla = Del()
u_field = (C.x**3)*(C.y**2)*(C.z) + C.x*sin(2*pi*C.x*C.z)*sin(2*pi*C.x*C.y)*sin(2*pi*C.z)
grad_u = nabla(u_field, doit=True)
u_func = lambdify([C.x, C.y, C.z], u_field)

K_sym = Matrix(([C.y**2 + C.z**2 + 1, -C.x*C.y, -C.x*C.z] ,
                [-C.x*C.y, C.x**2 + C.z**2 + 1, -C.y*C.z],
                [-C.x*C.z, -C.y*C.z, C.x**2 + C.y**2 + 1]))
K_func = lambda x, y, z: np.array([y**2 + z**2 + 1, -x*y, -x*z, \
                                -x*y, x**2 + z**2 + 1, -y*z, \
                                -x*z, -y*z, x**2 + y**2 + 1])

print("Generating mesh...")
mesh_gen = MeshGenerator('61052_vols.h5m')
vertices = mesh_gen.mb.get_entities_by_dimension(0, 0)
mesh_gen.mtu.construct_aentities(vertices)
mesh_gen.set_centroids()
mesh_gen.set_heterogeneous_permeability(K_func)

grad_u_matrix = Matrix([grad_u.coeff(C.i), grad_u.coeff(C.j), grad_u.coeff(C.k)])
F = K_sym * grad_u_matrix
div_F = F.diff(C.x)[0] + F.diff(C.y)[1] + F.diff(C.z)[2]
div_F_func = lambdify([C.x, C.y, C.z], div_F)
mesh_gen.set_source_term(div_F_func)

mesh_gen.set_dirichlet_boundary_conditions(u_func)

print("Done")

mesh_gen.mb.write_file('test_case_2_eymard.h5m')
