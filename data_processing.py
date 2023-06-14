import numpy as np
from utils import utility as ut


def create_features(data: dict, sae_params: dict) -> list:
    nclasses    = sae_params["nclasses"]
    X, Y        = np.array([]), np.array([])
    for idx, s in enumerate(data.values()):
        amplitudes  = get_amplitudes(s, sae_params)
        Y_binary    = create_binary_labels(idx, amplitudes.shape[0], nclasses)
        Y           = stack_arrays(Y, Y_binary)
        X           = stack_arrays(X, amplitudes)
    X = normalize_data(X)
    return X, Y

def normalize_data(X: np.ndarray, a=0.01, b=0.99):
    return ((X - X.min())/(X.max() - X.min())) * (b - a) + a

def get_amplitudes(s: np.ndarray, sae_params: dict):
    nframe      = sae_params["nframe"]
    frame_size  = sae_params["frame_size"]
    signals     = s.T[:, :nframe * frame_size].reshape(s.T.shape[0], nframe, frame_size)
    ft          = np.fft.fft(signals, axis=2)
    return np.abs(ft[:, :, :ft.shape[2]//2]).reshape(-1, ft.shape[2]//2)

    
def stack_arrays(arr: np.ndarray, new_arr: np.ndarray) -> np.ndarray:
    return np.concatenate((arr, new_arr)) if arr.shape[0] != 0 else new_arr


def save_data(features: list, p_training: float) -> None:
    file = "DATA/processed_data"
    X_train, Y_train, X_test, Y_test  = train_test_split(features, p_training)
    np.savetxt(f"{file}/X_train.csv", X_train, delimiter=",", fmt="%.4f")
    np.savetxt(f"{file}/X_test.csv", X_test, delimiter=",", fmt="%.4f")
    np.savetxt(f"{file}/Y_train.csv", Y_train, delimiter=",", fmt="%.4f")
    np.savetxt(f"{file}/Y_test.csv", Y_test, delimiter=",", fmt="%.4f")


def train_test_split(features: list, p: float) -> list:
    X, Y  = features[0], features[1]
    M     = np.concatenate((X, Y), axis = 1)
    np.random.shuffle(M)
    split_index       = int(M.shape[0] * p)
    trn_set, test_set = M[:split_index, :], M[split_index:, :]
    X_train, Y_train  = trn_set[:, : X.shape[1]], trn_set[:, -Y.shape[1]:]
    X_test, Y_test    = test_set[:, : X.shape[1]], test_set[:, -Y.shape[1]:]
    return X_train, Y_train, X_test, Y_test


def create_binary_labels(i: int, m: int, n: int) -> np.ndarray:
    binary_array       = np.zeros((m, n))
    binary_array[:, i] = 1
    return binary_array


def main():
    dae_params      = ut.load_dae_config("config/cnf_dae.csv")
    data            = ut.load_raw_data("DATA/origin_data/", dae_params["nclasses"])
    features        = create_features(data, dae_params)
    save_data(features, dae_params["p_training"])
    

if __name__ == '__main__': 
	main()


