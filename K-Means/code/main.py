import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from k_means import *

DATASET = "iris.csv"
K_NUM   = 3
EPOCH   = 100

clt = K_means(DATASET, K_NUM, EPOCH)
clt.train()
