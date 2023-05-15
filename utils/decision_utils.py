import jax.numpy as jnp
import pickle


def save(args, Nx, Ny, rmse_BMC, rmse_KMS, rmse_LSMC):
    rmse_dict = {}
    rmse_dict["BMC"] = rmse_BMC
    rmse_dict["KMS"] = rmse_KMS
    rmse_dict["LSMC"] = rmse_LSMC
    with open(f"{args.save_path}/rmse_dict_X_{Nx}_y_{Ny}", 'wb') as f:
        pickle.dump(rmse_dict, f)
    return


def scale(Z):
    s = Z.std()
    standardized = Z / s
    return standardized, s


def standardize(Z):
    mean = Z.mean(0)
    std = Z.std(0)
    standardized = (Z - mean) / std
    return standardized, mean, std
