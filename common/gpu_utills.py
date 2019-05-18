import warnings
from subprocess import Popen, PIPE


def get_num_gpu():
    try:
        p = Popen(["nvidia-smi",
                   "--query-gpu=index",
                   "--format=csv,noheader,nounits"], stdout=PIPE)

        stdout, _ = p.communicate()

        num_gpus = len(stdout.split(sep=b'\n')) - 1  # Num of GPU's is  the number of lines produced at stdout
    except:
        warnings.warn('Cannot find the number of available GPUs. Using 0.')
        num_gpus = 0

    return num_gpus
