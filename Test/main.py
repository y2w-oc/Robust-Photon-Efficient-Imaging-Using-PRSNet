# The test file for network
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from tensorboardX import SummaryWriter
from datetime import datetime
import skimage.io
import scipy.io as scio
from glob import glob
import pathlib

from pro.Fn_Test import test_sm_HR
from models import PRSNet
from util.ParseArgs import parse_args

    
def main():
    # parse arguments
    opt = parse_args("./config.ini")
    print("Number of available GPUs: {} {}".format(torch.cuda.device_count(), \
        torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Test Model Path: {}".format(opt["testModelsDir"]))
    print("Test Data Path: {}".format(opt["testDataDir"]))
    print("Test Output Path: {}".format(opt["testOutDir"]))
    # list all the test models
    file_list = sorted(glob(opt["testModelsDir"]))
    print("+++++++++++++++++++++++++++++++++++++++++++")

    # configure network
    model = PRSNet.PixelWiseResShrinkNet()
    model.cuda()
    model.eval()

    # test all the pretrained models
    for iter, pre_model in enumerate(file_list):

        print('The total number of test models are: {}'.format(len(file_list)))

        filename, _ = os.path.splitext(os.path.split(pre_model)[1])
        outdir_m = opt["testOutDir"] + '/Model_'+ filename
        pathlib.Path(outdir_m).mkdir(parents=True, exist_ok=True)

        print('=> Loading checkpoint {}'.format(pre_model))
        ckpt = torch.load(pre_model)
        model_dict = model.state_dict()

        try:
            ckpt_dict = ckpt["state_dict"]
        except KeyError:
            print('Key error loading state_dict from checkpoint; assuming checkpoint contains only the state_dict')
            ckpt_dict = ckpt

        # to update the model using the pretrained models
        for key_iter, k in enumerate(ckpt_dict.keys()):
            model_dict.update({k: ckpt_dict[k]})
            if key_iter == (len(ckpt_dict.keys()) - 1):
                print('Model Parameter Update!')

        model.load_state_dict(model_dict)

        # run batch-processing function
        rmse, runtime = test_sm_HR(model, opt, outdir_m)

        print("Model: {} and Performance: {} with run time of: {}".format(filename, rmse, runtime))



if __name__=="__main__":
    print("Start testing......")
    main()



