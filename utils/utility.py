import numpy as np

def load_dae_config(file_path: str) -> dict:
    cnf_dae = np.loadtxt(file_path, delimiter = ",")
    return {
        "nclasses"    : int(cnf_dae[0]),
        "nframe"      : int(cnf_dae[1]),
        "frame_size"  : int(cnf_dae[2]),
        "p_training"  : cnf_dae[3],
        "encoder_act" : int(cnf_dae[4]),
        "max_iter"    : int(cnf_dae[5]),
        "batch_size"  : int(cnf_dae[6]),
        "alpha"       : cnf_dae[7],
        "encoders"    : cnf_dae[8:],
    }

def load_softmax_config(file_path: str) -> dict:
    cnf_softmax = np.loadtxt(file_path, delimiter = ",")
    return {
        "max_iter"  : int(cnf_softmax[0]),
        "alpha"     : int(cnf_softmax[1]),
        "batch_size": int(cnf_softmax[2])
    }


def load_raw_data(file_name: str, n_classes: int)-> dict:
    return {
        f"class{i}": np.loadtxt(
            f"{file_name}/class{i}.csv", delimiter=",")
        for i in range(1, n_classes + 1)
    }