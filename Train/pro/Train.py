# The train function
import numpy as np 
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
import scipy.io as scio
from pro.Validate import validate
from util.SaveChkp import save_checkpoint
from pro.Loss import criterion_KL,criterion_TV,criterion_L2

cudnn.benchmark = True
lsmx = torch.nn.LogSoftmax(dim=1)
dtype = torch.cuda.FloatTensor
CELoss = torch.nn.CrossEntropyLoss()
C = 3e8
Tp = 80e-12

def train(model, train_loader, val_loader, optimer, epoch, n_iter,
            train_loss, val_loss, params, logWriter):
    for sample in tqdm(train_loader):
        # configure model state
        model.train()

        # load data and train the network
        M_mea = sample["spad"].type(dtype)
        dep = sample["bins"].type(dtype)

        M_mea_re, dep_re = model(M_mea)

        out = M_mea_re.unsqueeze(1)
        out = out.permute(0, 1, 3, 4, 2)
        out = out.contiguous().view(-1, 1024)
        dep = dep.long().view(-1)
        loss_ce = CELoss(out, dep)
        loss_tv = criterion_TV(dep_re)
        dep_re = dep_re.float() * Tp * C / 2
        dep = dep.view(-1, 1, 32, 32).float() * Tp * C / 2
        rmse = criterion_L2(dep_re, dep)

        loss = loss_ce + params["p_tv"]*loss_tv

        optimer.zero_grad()
        loss.backward()
        optimer.step()
        n_iter += 1

        logWriter.add_scalar("loss_train/all", loss, n_iter)
        logWriter.add_scalar("loss_train/ce", loss_ce, n_iter)
        logWriter.add_scalar("loss_train/tv", params["p_tv"]*loss_tv, n_iter)
        logWriter.add_scalar("loss_train/rmse", rmse, n_iter)
        train_loss["CE"].append(loss_ce.data.cpu().numpy())
        train_loss["TV"].append((params["p_tv"]*loss_tv).data.cpu().numpy())
        train_loss["RMSE"].append(rmse.data.cpu().numpy())

        if n_iter % params["save_every"] == 0:
            #print("Sart validation...")
            # val_loss, logWriter = validate(model, val_loader, n_iter, val_loss, params, logWriter)

            scio.savemat(file_name=params["log_file"]+"/train_loss.mat", mdict=train_loss)
            scio.savemat(file_name=params["log_file"]+"/val_loss.mat", mdict=val_loss)
            # save model states
            print("Saving checkpoint...")
            save_checkpoint(n_iter, epoch, model, optimer,
                file_path=params["log_file"]+"/epoch_{}_{}.pth".format(epoch, n_iter))
            print("Checkpoint saved!")
    
    return model, optimer, n_iter, train_loss, val_loss, logWriter

