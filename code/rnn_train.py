from rnn import *
from util import Util

import torch
import torch.nn as nn
from torch.autograd import Variable

import pickle
import html
from collections import Counter
import numpy as np
import os
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KBP Slot Filling RNN model')

    parser.add_argument("username", type = str, help = "Username")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-4, help = "Learning Rate for train")
    parser.add_argument("-dr", "--decay_rate", type = float, default = 0.9, help = "Decay Rate for train")
    parser.add_argument("-bs", "--batch_size", type = int, default = 500, help = "Batch Size for Train")
    parser.add_argument("--train_from", type = int, default = 0, help = "Train from model ('-1' will be train from last epoch)")

    parser.add_argument("--use_gpu", action = "store_true", default = False, help = "GPU usage flag")

    args = parser.parse_args()

    sentence = []
    entity = []
    entity_position = []
    filler = []
    filler_position = []
    relation = []

    for data_name in os.listdir("../result/%s/label" % args.username):
        with open("../result/%s/label/%s" % (args.username, data_name)) as f:
            line = f.read()
            while(line[-1] == "\n"):
                line = line[:-1]
            sent, e1, e2, e1_position, e2_position, rel, label = line.split("\t")
        e1_position = list(map(int, e1_position.split(",")))
        e2_position = list(map(int, e2_position.split(",")))
        label = int(label)

        if label == 1:
            sentence.append(sent)
            entity.append(e1)
            filler.append(e2)
            entity_position.append(e1_position)
            filler_position.append(e2_position)
            relation.append(rel)
   
    model_path = "../model/%s" % args.username
    classifier = torch.load("../model/%s/latest" % (args.username))

    while(True):
        classifier.


