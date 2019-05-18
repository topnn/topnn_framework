function calc_betti(filepath_betti, file_path_curve, file_repres, maxdim,  mat)
    print("\nsymmetric mat")
    print("\nstarted calculating betti numbers for mat")

    if isa(mat, Array)
        print("\nmat is a matrix")
        C = Eirene.eirene(mat, maxdim=maxdim, minrad=1, maxrad=2, numrad=1, record = "cyclerep")
    else
        print("\nmat is a file")
        mat = npzread(mat)
        maxdim = parse(Int64, maxdim)
        C = Eirene.eirene(mat, maxdim=maxdim, minrad=1, maxrad=2, numrad=1, record = "cyclerep")
    end

    print("\nDone calculating betti numbers\n")

    dim = maxdim
    while dim >=0
        t = Eirene.barcode(C, dim = dim)

        print("\nBetti neighbors filtration calc:\n")
        print(t)

        # save results to csv file.
        f = open(filepath_betti * "_" * string(dim),"w")

        print("\nstoring betti numbers in \n")
        print(filepath_betti)
        print("\n")

        for i in 1:length(t[:,1])
               write(f, @sprintf("%20.16f, %20.16f\n", t[i, 1], t[i,2]))
        end
        close(f)


        print("\nDone storing betti numbers \n")

        print("\n Extract betti curve \n")
        B = Eirene.betticurve(C, dim=dim)
        print("\nbetti curve\n")
        print(B)

        f = open(file_path_curve * "_" * string(dim), "w")
        print("\nstoring betti numbers in \n")
        print(file_path_curve)
        print("\n")

        for i in 1:length(B[:])
              write(f, @sprintf("%20.16f, %20.16f\n", i, B[i]))
        end

        close(f)
        print("\nDone, storing betti numbers in \n")
        print("\nDone all mat calculations.\n")

        print("\nExtracting representative.\n")

        f = open(file_repres  * "_" * string(dim), "w")
        print("\nstoring representatives in \n")
        print(file_repres)
        print("\n")

        for i in 1:length(t[:,1])
            S = Eirene.classrep(C, class=i, dim=dim)
            write(f, "$S\n")
        end

        close(f)
        print("\nDone, storing representatives " * string(dim)* " .\n\n\n\n\n\n")
        dim = dim - 1
    end
end