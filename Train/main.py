# The train file for network
# Based on pytorch 1.0
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms
import os
import sys
from tensorboardX import SummaryWriter
from datetime import datetime
import skimage.io
import scipy.io as scio

from util.SpadDataset import SpadDataset
from util.ParseArgs import parse_args
from util.SaveChkp import save_checkpoint
from pro.Train import train
from models import PRSNet

    
def main():
    
    # parse arguments
    opt = parse_args("./config.ini")
    print("Number of assigned GPUs: {}".format(opt["gpu_num"]))
    print("Number of available GPUs: {} {}".format(torch.cuda.device_count(), \
        torch.cuda.get_device_name(torch.cuda.current_device())))
    print("Batch_size: {}\n".format(opt["batch_size"]),
        "Workers: {}\n".format(opt["workers"]), 
        "Epoch: {}\n".format(opt["epoch"]), 
        "Save_every: {}\n".format(opt["save_every"]), 
        "Lri: {}\n".format(opt["lri"]), 
        "Optimizer: {}\n".format(opt["optimizer"]), 
        "Noise_idx: {}\n".format(opt["noise_idx"]),
        "Model_name: {}\n".format(opt["model_name"]),  
        "Log_dir: {}\n".format(opt["log_dir"]), 
        "Log_file: {}\n".format(opt["log_file"]), 
        "Util_dir: {}\n".format(opt["util_dir"]), 
        "Train_file: {}\n".format(opt["train_file"]), 
        "Val_file: {}\n".format(opt["val_file"]), 
        "Resume: {}\n".format(opt["resume"]), 
        "Resume_fpt: {}\n".format(opt["resume_fpt"]), 
        "Resume_mod: {}\n".format(opt["resume_mod"]), 
        "Train_loss: {}\n".format(opt["train_loss"]), 
        "Val_loss: {}".format(opt["val_loss"]))#, sep="")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
    # load data
    print("Loading training data...")
    # data preprocessing
    train_data = SpadDataset(opt["train_file"], opt["noise_idx"], 32)
    train_loader = DataLoader(train_data, batch_size=opt["batch_size"], 
                            shuffle=True, num_workers=opt["workers"], 
                            pin_memory=True)
    print("Load training data complete!\nLoading validation data...")
    val_data = SpadDataset(opt["val_file"], opt["noise_idx"], 32)
    val_loader = DataLoader(val_data, batch_size=opt["batch_size"], 
                            shuffle=True, num_workers=opt["workers"], 
                            pin_memory=True)
    print("Load validation data complete!")
    print("+++++++++++++++++++++++++++++++++++++++++++")

    # configure network
    print("Constructing Models...")
    model = PRSNet.PixelWiseResShrinkNet()
    model.cuda()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print("Models constructed complete!")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    
    # construct optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, opt['lri'])

    n_iter = 0
    start_epoch = 1
    train_loss = {"CE": [], "TV": [], "RMSE": []}
    val_loss = {"CE": [], "TV": [], "RMSE": []}
    logWriter = SummaryWriter(opt["log_file"] + "/")
    print("Parameters initialized")
    print("+++++++++++++++++++++++++++++++++++++++++++")

    if opt["resume"]:
        if os.path.exists(opt["resume_mod"]):
            print("Loading checkpoint from {}".format(opt["resume_mod"]))
            checkpoint = torch.load(opt["resume_mod"])
            
            # load start epoch
            try:
                start_epoch = checkpoint['epoch']
                print("Loaded and update start epoch: {}".format(start_epoch))
            except KeyError as ke:
                start_epoch = 1
                print("No epcoh info found in the checkpoint, start epoch from 1")
            
            # load iter number
            try:
                n_iter = checkpoint["n_iter"]
                print("Loaded and update start iter: {}".format(n_iter))
            except KeyError as ke:
                n_iter = 0
                print("No iter number found in the checkpoint, start iter from 0")

            # load learning rate
            try:
                opt["lri"] = checkpoint["lr"]
            except KeyError as ke:
                print("No learning rate info found in the checkpoint, use initial learning rate:")
            
            # load model params
            model_dict = model.state_dict()
            try:
                ckpt_dict = checkpoint['state_dict']
                for k in ckpt_dict.keys():
                    model_dict.update({k[7:]: ckpt_dict[k]})
                model.load_state_dict(model_dict)
                print("Loaded and update model states!")
            except KeyError as ke:
                print("No model states found!")
                sys.exit("NO MODEL STATES")

            # load optimizer state
            for g in optimizer.param_groups:
                g["lr"] = opt["lri"]
            print("Loaded learning rate!")
               
            # load mat files
            try:
                train_loss["CE"] = list(scio.loadmat(opt["train_loss"])["CE"])
                train_loss["TV"] = list(scio.loadmat(opt["train_loss"])["TV"])
                train_loss["RMSE"] = list(scio.loadmat(opt["train_loss"])["RMSE"])
                val_loss["CE"] = list(scio.loadmat(opt["val_loss"])["CE"])
                val_loss["TV"] = list(scio.loadmat(opt["val_loss"])["TV"])
                val_loss["RMSE"] = list(scio.loadmat(opt["val_loss"])["RMSE"])
                print("Loaded and update train and val loss from assigned path!")
            except FileNotFoundError as fnf:
                print("No train or val loss mat found.\nUse initial ZERO")
            
            print("Checkpoint load complete!!!")

        else:
            print("No checkPoint found at {}!!!".format(opt["resume_mod"]))
            sys.exit("NO FOUND CHECKPOINT ERROR!")

    else:
        print("Do not resume! Use initial params and train from scratch.")

    # start training 
    print("Start training...")
    for epoch in range(start_epoch, opt["epoch"]):
        print("Epoch: {}, LR: {}".format(epoch, optimizer.param_groups[0]["lr"]))

        Mod_Dict, optimizer, n_iter, train_loss, val_loss, logWriter = \
            train(model, train_loader, val_loader, optimizer, \
                epoch, n_iter, train_loss, val_loss, opt, logWriter)

        print("==================>Train<==================")
        print("CE: {}, TV: {}, RMSE: {}".format(\
            np.mean(train_loss["CE"][-(len(train_data)-1):]), \
                np.mean(train_loss["TV"][-(len(train_data)-1):]), \
                    np.mean(train_loss["RMSE"][-(len(train_data)-1):])))
        print("==================>Validation<==================")
        print("CE: {}, TV: {}, RMSE: {}".format(\
            np.mean(val_loss["CE"][-(len(train_data)//(opt["batch_size"]*opt["save_every"])-1):]), \
                np.mean(val_loss["TV"][-(len(train_data)//(opt["batch_size"]*opt["save_every"])-1):]), \
                    np.mean(val_loss["RMSE"][-(len(train_data)//(opt["batch_size"]*opt["save_every"])-1):])))

        # lr update 
        for g in optimizer.param_groups:
            g["lr"] *= .6
        
        # save checkpoint every epoch
        save_checkpoint(n_iter, epoch, model, optimizer,\
            file_path=opt["log_file"]+"/epoch_{}_{}_END.pth".format(epoch, n_iter))

        print("End of epoch: {}. Checkpoint saved!".format(epoch))


if __name__=="__main__":
    print("Start training...")
    main()
