# load Eirene library
include("C:\\topology_prj\\framework_repo_f\\topology_of_dl\\julia\\Eirene.jl\\src\\Eirene.jl")

# Substitutet "Eirene.jl\src\examples\test.csv" csv file with 
# the desired point cloud data and use built in "eirenefilepath" method to load that csv
# filepath = Eirene.eirenefilepath("test")
filepath = joinpath(@__DIR__,"examples/test.csv")
# read csv
pointcloud = Eirene.readcsv(filepath)

# evaluate persitant homology of dimensions up to "maxdim", where persistance starts at 
# radius "minrad" up to radius "maxrad" and is evaluated at "numrad" equaly spaces points.
C = Eirene.eirene(pointcloud,maxdim=1, minrad=0.15, maxrad=0.8, numrad=2, model="pc")

# display peristance intervals of dimension "dim"
t = Eirene.barcode(C, dim = 0)

# save results to csv file.
f = open("C:\\topology_prj\\framework_repo_f\\topology_of_dl\\julia\\bars.csv","w")
for i in 1:length(t[:,1])
       write(f, @sprintf("%20.16f, %20.16f\n", t[i, 1], t[i,2]))
end
close(f)