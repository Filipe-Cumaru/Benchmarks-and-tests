from mesh_generator_ricardo import GenerateMesh

for i in range(1, 9):
    in_mesh_file = "tet_" + str(i) + ".msh"
    out_mesh_file = "tet_" + str(i) + ".h5m"
    mesh_gen = GenerateMesh(in_mesh_file)
    mesh_gen.get_all_vertices()
    mesh_gen.create_volumes()
    mesh_gen.mb.write_file(out_mesh_file)

