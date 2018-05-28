from util import Util
from rnn import *

import os
import signal
import subprocess
import html
import pickle

from bokeh.plotting import figure

USE_GPU = False

class user():
    def __init__(self):
        self.is_active = False
        self.name = "Unknown"

    def login(self, name, password):
        if name not in os.listdir("../user/"):
            return 0, "Wrong User Name."

        with open("../user/%s/user_info" % name) as f:
            info = f.read()
            while(info[-1] == "\n"):
                info = info[:-1]

        if info == password:
            self.is_active = True
            self.name = name
            with open("../user/%s/cursor" % name) as f2:
                self.cursor = int(f2.read())
            with open("../user/%s/pid" % name) as f2:
                self.pid = int(f2.read())

            with open("../user/%s/result/util.pkl" % name, "rb") as f2:
                self.util = pickle.load(f2)

            with open("../user/%s/model_info.pkl" % name, "rb") as f2:
                self.model_info = pickle.load(f2)
            if os.path.exists("../user/%s/model/%d_%d_%d_latest" % (name, self.model_info["word_embedding_dim"], self.model_info["position_embedding_dim"], self.model_info["sentence_embedding_dim"])) is False:
                word_emb = Word_Embedding(self.model_info["word_embedding_dim"], self.util, USE_GPU)
                entity_position_emb = Position_Embedding(self.model_info["position_embedding_dim"], self.util, USE_GPU)
                filler_position_emb = Position_Embedding(self.model_info["position_embedding_dim"], self.util, USE_GPU)
                sentence_emb = Sentence_Embedding(word_emb, entity_position_emb, filler_position_emb, self.util, USE_GPU)

                classifier = Classifier(sentence_emb, self.model_info["sentence_embedding_dim"], self.util, USE_GPU)

                classifier.save(name)

            return 1, "Success."

        return 0, "Wrong Password."

    def register(self, name, password):
        if name in os.listdir("../user/"):
            return 0, "The User Name already exist."

        os.makedirs("../user/%s/" % name)

        with open("../user/%s/user_info" % name, "w") as f:
            f.write(password)
        with open("../user/%s/cursor" % name, "w") as f:
            f.write("0")
        with open("../user/%s/pid" % name, "w") as f:
            f.write("-1")
        with open("../user/%s/model_info.pkl" % name, "wb") as f:
            pickle.dump({"word_embedding_dim":150,
                "position_embedding_dim":25,
                "sentence_embedding_dim":200}, f)

        os.makedirs("../user/%s/model" % name)

        os.makedirs("../user/%s/figure" % name)
        os.makedirs("../user/%s/figure/data_vis" % name)
        os.makedirs("../user/%s/figure/model_vis" % name)

        os.makedirs("../user/%s/result" % name)
        os.makedirs("../user/%s/result/label/" % name)
        os.makedirs("../user/%s/result/train/" % name)
        os.makedirs("../user/%s/result/valid/" % name)

        self.set_data(name, "ANGELIS_POSITION.pkl", "./relation_list.txt")

        return 1, "Success."

    def load_data(self, data_type, cursor):
        with open("../user/%s/result/%s/%d" % (self.name, data_type, cursor)) as f:
            line = f.read()
        while(line[-1] == "\n"):
            line = line[:-1]
        sent, e1, e2, e1_position, e2_position, rel, label = line.split("\t")
        e1_position = list(map(int, e1_position.split(",")))
        e2_position = list(map(int, e2_position.split(",")))
        label = int(label)

        return sent, e1, e2, e1_position, e2_position, rel, label

    def set_data(self, username, data_name, relation_dict_path):
        with open(relation_dict_path) as f:
            tmp = f.read().split("\n")

        if tmp[-1] == "":
            tmp = tmp[:-1]

        relation_list = []

        for i in tmp:
            if i[0] != "#":
                relation_list.append(i)

        relation_dict = {relation:i for i, relation in enumerate(relation_list)}

        with open("../data/%s" % data_name,"rb") as f:
            data = pickle.load(f)

        sentence = []
        entity = []
        entity_position = []
        filler = []
        filler_position = []
        relation = []

        for sent, ent, entity_begin, entity_end, fil, filler_begin, filler_end, rel, confidence in data:
            #if (rel != "no_relation") & (rel not in relation_dict):
            if (rel not in relation_dict):
                continue
            sentence.append(html.unescape(sent.lower()))

            splited_sent = html.unescape(sent.lower()).split(" ")

            entity.append(html.unescape(ent.strip()).lower())
            filler.append(html.unescape(fil.strip()).lower())

            tmp_ent_pos = []
            for i in range(len(splited_sent)):
                if i < entity_begin:
                    tmp_ent_pos.append(i - entity_begin)
                elif i < (entity_end - 1):
                    tmp_ent_pos.append(0)
                else:
                    tmp_ent_pos.append(i - (entity_end - 1))
            entity_position.append(tmp_ent_pos)

            tmp_fil_pos = []
            for i in range(len(splited_sent)):
                if i < filler_begin:
                    tmp_fil_pos.append(i - filler_begin)
                elif i < (filler_end - 1):
                    tmp_fil_pos.append(0)
                else:
                    tmp_fil_pos.append(i - (filler_end - 1))

            filler_position.append(tmp_fil_pos)

            relation.append(rel)

        print("# of data:", len(sentence))

        VALID_START = int(0.7 * len(sentence))

        util = Util(sentence[:VALID_START], "./relation_list.txt", 3, 10)
        with open("../user/%s/result/util.pkl" % (username), "wb") as f:
            pickle.dump(util, f)

        for idx in range(len(sentence)):
            if idx < VALID_START:
                data_type = "train"
                data_idx = idx
            else:
                data_type = "valid"
                data_idx = idx - VALID_START
            with open("../user/%s/result/%s/%d" % (username, data_type, data_idx), "w") as f:
                f.write(sentence[idx])
                f.write("\t")
                f.write(entity[idx])
                f.write("\t")
                f.write(filler[idx])
                f.write("\t")
                f.write(",".join(map(str, entity_position[idx])))
                f.write("\t")
                f.write(",".join(map(str, filler_position[idx])))
                f.write("\t")
                f.write(relation[idx])
                f.write("\t")
                f.write(str(1))

    def update(self, sentence, e1, e2, e1_position, e2_position, relation, label):
        with open("../user/%s/result/label/%d" % (self.name, self.cursor), "w") as f:
            f.write(sentence)
            f.write("\t")
            f.write(e1)
            f.write("\t")
            f.write(e2)
            f.write("\t")
            f.write(",".join(map(str, e1_position)))
            f.write("\t")
            f.write(",".join(map(str, e2_position)))
            f.write("\t")
            f.write(relation)
            f.write("\t")
            f.write(str(label))

        self.cursor += 1
        with open("../user/%s/cursor" % self.name, "w") as f:
            f.write(str(self.cursor))

    def statistics(self):
        stat_dict = dict()
        for rel in self.util.relation_dict.keys():
            rel = rel[4:]
            stat_dict[rel] = [0, 0]
        #stat_dict["no_relation"] = [0, 0]

        for data_name in os.listdir("../user/%s/result/label/" % self.name):
            with open("../user/%s/result/label/%s" % (self.name, data_name)) as f:
                line = f.read()
            while(line[-1] == "\n"):
                line = line[:-1]
            sent, e1, e2, e1_position, e2_position, rel, label = line.split("\t")
            label = int(label)

            stat_dict[rel][label] += 1

        return stat_dict

    def loss_graph(self):
        with open("../user/%s/model/loss" % self.name) as f:
                loss_list = f.read().split("\n")

        while(loss_list[-1] == ""):
            loss_list = loss_list[:-1]

        p = figure(plot_width=400, plot_height=150)
        p.line(list(range(len(loss_list))), loss_list, line_width=1)

        return p

    def run_classifier(self):
        p = subprocess.Popen(["python3 rnn_train.py"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open("../user/%s/pid" % self.name, "w") as f:
            f.write(str(p.pid))

        self.pid = p.pid
