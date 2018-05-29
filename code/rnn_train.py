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
    parser.add_argument("word_embedding_size", type = int, help = "The size of word embedding model")
    parser.add_argument("position_embedding_size", type = int, help = "The size of position embedding model")
    parser.add_argument("hidden_size", type = int, help = "The size of hidden of LSTM")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-1, help = "Learning Rate for train")
    parser.add_argument("-dr", "--decay_rate", type = float, default = 0.9, help = "Decay Rate for train")
    parser.add_argument("-bs", "--batch_size", type = int, default = 500, help = "Batch Size for Train")
    parser.add_argument("--train_from", type = int, default = -1, help = "Train from model ('-1' will be train from last epoch)")

    parser.add_argument("--use_gpu", action = "store_true", default = False, help = "GPU usage flag")

    args = parser.parse_args()

    model_path = "../model/%s" % args.username
    if args.train_from == -1:
        classifier = torch.load("../user/%s/model/%d_%d_%d_latest" % (args.username, args.word_embedding_size, args.position_embedding_size, args.hidden_size))

    while(True):
        sentence = []
        entity = []
        entity_position = []
        filler = []
        filler_position = []
        relation = []
        
        label_cnt = 0
        for data_name in os.listdir("../user/%s/result/label" % args.username):
            with open("../user/%s/result/label/%s" % (args.username, data_name)) as f:
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
                label_cnt += 1
   
        # 고치자 로스 부분 수저앻야함
        classifier.train(sentence, entity_position, filler_position, relation, args.batch_size, epoch = 1, learning_rate = args.learning_rate, username = args.username)
        continue

        sentence = []
        entity = []
        entity_position = []
        filler = []
        filler_position = []
        relation = []

        for data_name in os.listdir("../user/%s/result/train" % args.username):
            with open("../user/%s/result/train/%s" % (args.username, data_name)) as f:
                line = f.read()
                while(line[-1] == "\n"):
                    line = line[:-1]
                sent, e1, e2, e1_position, e2_position, rel, label = line.split("\t")
            e1_position = list(map(int, e1_position.split(",")))
            e2_position = list(map(int, e2_position.split(",")))
            label = int(label)

            sentence.append(sent)
            entity.append(e1)
            filler.append(e2)
            entity_position.append(e1_position)
            filler_position.append(e2_position)
            relation.append(rel[4:])
            #고치자 관계 안한거는 ?같은걸로 해서 컬로코딩
        classifier.visualize_data(sentence, entity_position, filler_position, relation, args.batch_size, args.username)
        exit()


