# update link to point to local Eirene library installation
include("/julia_include/Eirene_jl/src/Eirene.jl")
using NPZ

fn2 = include("./julia_aux2.jl")

neihbors_betti_file = ARGS[1]
neihbors_curve_file = ARGS[2]
file_repres_file = ARGS[3]
maxdim = ARGS[4]
mat_file = ARGS[5]

calc_betti(neihbors_betti_file,neihbors_curve_file,file_repres_file, maxdim ,mat_file)
exit()
