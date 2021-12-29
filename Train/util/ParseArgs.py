# The parse argments
import os
import sys
import configparser
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime

def parse_args(config_path = "./config.ini"):
    if os.path.exists(config_path):
        print("Reading config file from {} and parse args...".format(config_path))
        opt = {}
        config = ConfigParser(interpolation=ExtendedInterpolation())     # use ConfigParser realize a instance
        config.read(config_path)    # read the config file
        config_bk = ConfigParser()
        
        # get returns the "str" type
        opt["gpu_num"] = config.getint("params", "gpu_num")
        opt["batch_size"] = config.getint("params", "batch_size")
        opt["workers"] = config.getint("params", "workers")
        opt["epoch"] = config.getint("params", "epoch")
        opt["save_every"] = config.getint("params", "save_every")
        opt["optimizer"] = config.get("params", "optimizer")
        opt["lri"] = config.getfloat("params", "lri")
        opt["p_tv"] = config.getfloat("params", "p_tv")

        opt["noise_idx"] = config.getint("params", "noise_idx")
        opt["model_name"] = config.get("params", "model_name")
        opt["log_dir"] = config.get("params", "log_dir")
        opt["log_file"] = config.get("params", "log_file") + "_date_" + \
            datetime.now().strftime("%m_%d-%H_%M")

        opt["util_dir"] = config.get("params", "util_dir")
        opt["train_file"] = config.get("params", "train_file")
        opt["val_file"] = config.get("params", "val_file")
        opt["resume"] = config.getboolean("params", "resume")
        if opt["resume"]:
            opt["resume_fpt"] = config.get("params", "resume_fpt")
            opt["resume_mod"] = config.get("params", "resume_mod")
            opt["train_loss"] = config.get("params", "train_loss")
            opt["val_loss"] = config.get("params", "val_loss")
            opt["log_file"] += "_RESUME"
        else:
            opt["resume_fpt"] = "NONE"
            opt["resume_mod"] = "NONE"
            opt["train_loss"] = "NONE"
            opt["val_loss"] = "NONE"

        config_bk.read_dict({"params_bk": opt})
        config_bk_pth = opt["log_file"] + "/config_bk.ini"
        if not os.path.exists(opt["log_file"]):
            os.makedirs(opt["log_file"])
        with open(config_bk_pth, "w") as cbk_pth:
            config_bk.write(cbk_pth)
        
        print("Config file load complete! \nNew file saved to {}".format(config_bk_pth))
        return opt
    else:
        print("No file exist named {}".format(config_path))
        sys.exit("NO FILE ERROR")
