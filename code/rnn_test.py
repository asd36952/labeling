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

    args = parser.parse_args()

    try:
        with open("../user/%s/model/loss" % args.username) as f:
            loss_list = f.read().split("\n")
        while(loss_list[-1] == ""):
            loss_list = loss_list[:-1]
    except FileNotFoundError:
        loss_list = []

    with open("../user/%s/model/data_update" % args.username, "a") as f:
        f.write(str(len(loss_list)))
        f.write("\n")

    model_path = "../model/%s" % args.username
    if args.train_from == -1:
        classifier = torch.load("../user/%s/model/%d_%d_%d_latest" % (args.username, args.word_embedding_size, args.position_embedding_size, args.hidden_size))

    train_sentence = []
    train_entity = []
    train_entity_position = []
    train_filler = []
    train_filler_position = []
    train_relation = []

    for data_name in sorted(os.listdir("../user/%s/result/train" % args.username), key = lambda x:int(x)):
        with open("../user/%s/result/train/%s" % (args.username, data_name)) as f:
            line = f.read()
            while(line[-1] == "\n"):
                line = line[:-1]
            sent, e1, e2, e1_position, e2_position, rel, label = line.split("\t")
        e1_position = list(map(int, e1_position.split(",")))
        e2_position = list(map(int, e2_position.split(",")))
        label = int(label)

        train_sentence.append(sent)
        train_entity.append(e1)
        train_filler.append(e2)
        train_entity_position.append(e1_position)
        train_filler_position.append(e2_position)
        train_relation.append(rel[4:])

    valid_sentence = []
    valid_entity = []
    valid_entity_position = []
    valid_filler = []
    valid_filler_position = []
    valid_relation = []

    for data_name in sorted(os.listdir("../user/%s/result/valid" % args.username), key = lambda x:int(x)):
        with open("../user/%s/result/valid/%s" % (args.username, data_name)) as f:
            line = f.read()
            while(line[-1] == "\n"):
                line = line[:-1]
            sent, e1, e2, e1_position, e2_position, rel, label = line.split("\t")
        e1_position = list(map(int, e1_position.split(",")))
        e2_position = list(map(int, e2_position.split(",")))
        label = int(label)

        valid_sentence.append(sent)
        valid_entity.append(e1)
        valid_filler.append(e2)
        valid_entity_position.append(e1_position)
        valid_filler_position.append(e2_position)
        valid_relation.append(rel[4:])

    while(True):
        sentence = []
        entity = []
        entity_position = []
        filler = []
        filler_position = []
        relation = []
        label = []
        
        for data_name in sorted(os.listdir("../user/%s/result/label" % args.username), key = lambda x:int(x)):
            with open("../user/%s/result/label/%s" % (args.username, data_name)) as f:
                line = f.read()
                while(line[-1] == "\n"):
                    line = line[:-1]
                sent, e1, e2, e1_position, e2_position, rel, lab = line.split("\t")
            e1_position = list(map(int, e1_position.split(",")))
            e2_position = list(map(int, e2_position.split(",")))
            lab = int(lab)

            sentence.append(sent)
            entity.append(e1)
            filler.append(e2)
            entity_position.append(e1_position)
            filler_position.append(e2_position)
            relation.append(rel)
            label.append(lab)
   
        classifier.train(sentence, entity_position, filler_position, relation, label, args.batch_size, learning_rate = args.learning_rate, username = args.username)

        classifier.test(valid_sentence, valid_entity_position, valid_filler_position, valid_relation, args.batch_size, args.username)
        
        temp_relation = []
        for idx in range(len(label)):
            if label[idx] == 1:
                temp_relation.append(relation[idx])
            else:
                temp_relation.append("unknown")

        classifier.visualize_data(train_sentence, train_entity_position, train_filler_position, temp_relation + (["unlabeled"] * (len(train_relation) - len(temp_relation))), args.batch_size, args.username)

        classifier.visualize_model(train_sentence, train_entity_position, train_filler_position, args.batch_size, username = args.username)

        classifier.save(args.username)


