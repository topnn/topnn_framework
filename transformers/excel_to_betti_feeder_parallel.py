from transformers import Transformer
from os import listdir
import os
from os.path import isfile, join
from transformers.betti_calc_parallel import BettiCalcParallel
import csv
import multiprocessing
import datetime
import time
import psutil

def run_calculation(csv_file, task, model_path, julia, divisor, neighbors):
    print("Starting task ", task)
    loops = []
    betti_calc = BettiCalcParallel(model_path, julia, divisor, neighbors)    
    trace = betti_calc.transform(csv_file)

    loops.append({'trace': trace, 'name': os.path.basename(csv_file)})
    print("Done task ", task)
    return loops


class ExcelToBettiFeederParallel(Transformer):

    def __init__(self, model_path, julia,  excel_names, divisor, neighbors, max_dim):
        self.excel_dir = os.path.join(model_path, 'excels')
        self.model_path = model_path
        self.excel_names = excel_names
        self.julia = julia
        self.divisor = int(divisor)
        self.neighbors = int(neighbors)
        self.max_dim = int(max_dim)

    def transform(self, content=None):
        now = datetime.datetime.now()

        workers = 10
        giga = 1000000000
        mem_reserved = 10 * giga # reserve 10Gb of memory

        def split(arr, size):
            arrs = []
            while len(arr) > size:
                pice = arr[:size]
                arrs.append(pice)
                arr = arr[size:]
            arrs.append(arr)
            return arrs


        from threading import Thread
        loops = []

        subarrays = split(self.excel_names, workers)

        def check_threds(threads):
            alive_count = 0
            for thread in threads:
                if thread.is_alive():
                    alive_count += 1
            print("Currently", alive_count, "active threads", "out of max", workers, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            return alive_count

        def check_mem():
            mem = psutil.virtual_memory().free
            print("free mem (Gb):", 1.0 * psutil.virtual_memory().free / giga, "the reserved lower limit is (Gb) :",1.0 * mem_reserved/giga,  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return mem


        mat_files = []
        threads = []
        total = 0
        for i, subarray in enumerate(subarrays):
            for j, file in enumerate(subarray):

                print("running chunk", i)
                print("Submiting job", j)
                print("Total job count", total)
                csv_file = os.path.join(self.excel_dir, file)
                print("running on csv_file", csv_file)
                betti_calc = BettiCalcParallel(self.model_path, self.julia, self.divisor, self.neighbors, self.max_dim )
                (call, mat_file) = betti_calc.transform(csv_file)
                mat_files.append(mat_file)
                print("calling", call)
                print(":-:" * 30)

		# update os.system with path to julia executable
                threads.append(Thread(group=None, target=lambda: os.system('julia-9d11f62bcb/bin/' + call + " > " + os.path.join(self.model_path, "job-" + str(total) + ".txt"))))
                threads[total].start()
                print(":-:" * 30)
                total += 1

                while (check_threds(threads) == workers) or (check_mem() < mem_reserved):
                    print("All workers are busy or memory is low (Gb)", 1.0 * check_mem()/giga, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    time.sleep(10)

            print("done chunk ", i, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        while check_threds(threads) > 0 or (check_mem() < mem_reserved):
              time.sleep(10)

        print("all workers done")
        for mat_file in mat_files:
            os.remove(mat_file)

        print("done betti calculation for model", self.model_path , "in ", datetime.datetime.now() - now, " time")
        return ''
