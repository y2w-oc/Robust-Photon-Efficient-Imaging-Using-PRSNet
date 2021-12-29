import numpy as np
import torch
import torch.nn as nn
from glob import glob
import pathlib
import scipy
import os
import scipy.io as scio
import time
import h5py
from torchsummary import summary
from ptflops import get_model_complexity_info

smax = torch.nn.Softmax2d()
dtype = torch.cuda.FloatTensor


# batch processing function
###############################################################################################
def test_sm_HR(model, opt, outdir_m):

    rmse_all = []
    time_all = []
    print(opt["testDataDir"])
    root_path = opt["testDataDir"]
    for name_test in glob(root_path + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])

        # test_sm_crop_fast for simulated data
        # test_real_crop_fast for real world data
        ###############################################################################################
        t_all, rmse = test_sm_crop_fast(model, root_path, name_test_id, outdir_m)
        # t_all, rmse = test_real_crop_fast(model, root_path, name_test_id, outdir_m)
        time_all.append(t_all)
        rmse_all.append(rmse)

    return np.mean(rmse_all), np.mean(time_all)


def test_sm_crop_fast(model, root_path, name_test_id, outdir_m):
    name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"
    name_test = root_path + name_test_id + ".mat"
    print(name_test)
    print(name_test_save)

    dep = np.asarray(scio.loadmat(name_test)['depth']).astype(np.float32)
    h, w = dep.shape
    M_mea = scio.loadmat(name_test)["spad"]
    M_mea = scipy.sparse.csc_matrix.todense(M_mea)
    M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, w, h, -1])
    M_mea = np.transpose(M_mea, (0, 1, 4, 3, 2))

    M_mea = torch.from_numpy(M_mea)

    dep_rec_HR = np.zeros((h, w), dtype=np.float32)

    h_lim = (h-64) // 64
    w_lim = (w-64) // 64

    t_s = time.time()
    with torch.no_grad():
        for i in range(h_lim):
            for j in range(w_lim):
                print(i, j)
                mea = M_mea[:, :, :, i * 64: i * 64 + 128, j * 64: j * 64 + 128].type(dtype)
                dep_re = model(mea)
                dep_rec_HR[i * 64 + 32: i * 64 + 96, j * 64 + 32: j * 64 + 96] = dep_re.data.cpu().numpy()[0, 0, 32: 96, 32: 96]
                if i == 0:
                    dep_rec_HR[0: 32, j * 64: j * 64 + 128] = dep_re.data.cpu().numpy()[0, 0, 0: 32, :]
                if i == (h_lim - 1):
                    dep_rec_HR[h - 32: h, j * 64: j * 64 + 128] = dep_re.data.cpu().numpy()[0, 0, 96: 128, :]
                if j == 0:
                    dep_rec_HR[i * 64: i * 64 + 128, 0: 32] = dep_re.data.cpu().numpy()[0, 0, :, 0: 32]
                if j == (w_lim - 1):
                    dep_rec_HR[i * 64: i * 64 + 128, w-32: w] = dep_re.data.cpu().numpy()[0, 0, :, 96: 128]
                del mea
                del dep_re

    t_e = time.time()
    t_all = t_e - t_s

    C = 3e8
    Tp = 80e-12

    dist = dep_rec_HR * Tp * C / 2 + 0.012 # 0.012 compensates the index gap between python and matlab
    rmse = np.sqrt(np.mean((dist - dep) ** 2))

    scio.savemat(name_test_save, {"img_smax": dist, "rmse_smax": rmse})
    print("The RMSE: {}".format(rmse))

    return t_all, rmse


def test_real_crop_fast(model, root_path, name_test_id, outdir_m):
    name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"
    name_test = root_path + name_test_id + ".mat"
    print(name_test)

    M_mea = scio.loadmat(name_test)["spad"]
    M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, 256, 256, -1])
    M_mea = np.transpose(M_mea, (0, 1, 4, 3, 2))

    M_mea = torch.from_numpy(M_mea).type(dtype)

    dep_rec_HR = np.zeros((256, 256), dtype=np.float32)

    t_s = time.time()
    with torch.no_grad():
        for i in range(3):
            for j in range(3):
                print(i, j)
                mea = M_mea[:, :, :, i * 64: i * 64 + 128, j * 64: j * 64 + 128]
                dep_re = model(mea)
                dep_rec_HR[i * 64 + 32: i * 64 + 96, j * 64 + 32: j * 64 + 96] = dep_re.data.cpu().numpy()[0, 0, 32: 96, 32: 96]
                if i == 0:
                    dep_rec_HR[0: 32, j * 64: j * 64 + 128] = dep_re.data.cpu().numpy()[0, 0, 0: 32, :]
                if i == 2:
                    dep_rec_HR[224: 256, j * 64: j * 64 + 128] = dep_re.data.cpu().numpy()[0, 0, 96: 128, :]
                if j == 0:
                    dep_rec_HR[i * 64: i * 64 + 128, 0: 32] = dep_re.data.cpu().numpy()[0, 0, :, 0: 32]
                if j == 2:
                    dep_rec_HR[i * 64: i * 64 + 128, 224: 256] = dep_re.data.cpu().numpy()[0, 0, :, 96: 128]
                del mea
                del dep_re

    t_e = time.time()
    t_all = t_e - t_s

    C = 3e8
    Tp = 52e-12

    dist = dep_rec_HR * Tp * C / 2

    scio.savemat(name_test_save, {"img_smax": dist})

    return t_all, 0

