import numpy as np 
import torch
from tqdm import tqdm
from pro.Loss import criterion_KL,criterion_TV,criterion_L2

lsmx = torch.nn.LogSoftmax(dim=1)
dtype = torch.cuda.FloatTensor

CELoss = torch.nn.CrossEntropyLoss()
C = 3e8
Tp = 80e-12

def validate(model, val_loader, n_iter, val_loss, params, logWriter):

    model.eval()
    
    l_all = []
    l_ce = []
    l_tv = []
    l_rmse = []

    for sample in tqdm(val_loader):
        M_mea = sample["spad"].type(dtype)
        dep = sample["bins"].type(dtype)

        with torch.no_grad():
            M_mea_re, dep_re = model(M_mea)

        out = M_mea_re.unsqueeze(1)
        out = out.permute(0, 1, 3, 4, 2)
        out = out.contiguous().view(-1, 1024)
        dep = dep.long().view(-1)
        loss_ce = CELoss(out, dep).data.cpu().numpy()
        loss_tv = criterion_TV(dep_re).data.cpu().numpy()
        dep_re = dep_re.float() * Tp * C / 2
        dep = dep.view(-1, 1, 32, 32).float() * Tp * C / 2
        rmse = criterion_L2(dep_re, dep).data.cpu().numpy()

        loss = loss_ce + params["p_tv"]*loss_tv

        l_all.append(loss)
        l_ce.append(loss_ce)
        l_tv.append(loss_tv)
        l_rmse.append(rmse)

    # log the val losses
    logWriter.add_scalar("loss_val/all", np.mean(l_all), n_iter)
    logWriter.add_scalar("loss_val/cd", np.mean(loss_ce), n_iter)
    logWriter.add_scalar("loss_val/tv", np.mean(l_tv), n_iter)
    logWriter.add_scalar("loss_val/rmse", np.mean(l_rmse), n_iter)
    val_loss["CE"].append(np.mean(l_ce))
    val_loss["TV"].append(np.mean(l_tv))
    val_loss["RMSE"].append(np.mean(l_rmse))

    return val_loss, logWriter

