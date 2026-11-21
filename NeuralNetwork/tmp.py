import numpy as np
import pandas as pd
from IPython import embed
import matplotlib
matplotlib.use('QtAgg')   # 或 'TkAgg'/'Qt5Agg' 需要安装PyQt5或PyQt6 / 'MacOSX'
import matplotlib.pyplot as plt
from utils.MyTools import MyTools as mt
from utils.MyTf import RegHook

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F





if __name__ == "__main__":
    # tst_last_linear()
    # tst_attn()
    #获取encodelayer的注意力：
    # tst_attn2()
    
    
    
    embed()