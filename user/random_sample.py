import numpy as np
from shutil import copyfile

for user_idx in range(5):
    for data_idx in np.random.choice((743 - 435), 100, replace=False):
        copyfile("./random_%d/result/train/%d" % (user_idx + 1, data_idx + 435), "./random_%d/result/label/%d" % (user_idx + 1, data_idx + 435))
