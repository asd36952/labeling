from util import Util
from rnn import *

import os
import signal
import subprocess
import html
import pickle
import itertools

from bokeh.palettes import Category20 as palette
from bokeh.plotting import figure
from bokeh.models import Legend, TapTool, HoverTool, ColumnDataSource, OpenURL

USE_GPU = True

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
            if self.pid != -1:
                self.stop_classifier()

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
            pickle.dump({"word_embedding_dim":75,
                "position_embedding_dim":15,
                "sentence_embedding_dim":100}, f)

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

        TRAIN_START = int(0.4 * len(sentence))
        VALID_START = int(0.7 * len(sentence))
        print("Train Start:", TRAIN_START)
        print("Valid Start:", VALID_START)

        util = Util(sentence[:VALID_START], "./relation_list.txt", 3, 10)
        with open("../user/%s/result/util.pkl" % (username), "wb") as f:
            pickle.dump(util, f)

        for idx in range(len(sentence)):
            if idx < TRAIN_START:
                data_type = "label"
                data_idx = idx
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
                    f.write(relation[idx][4:])
                    f.write("\t")
                    f.write(str(1))

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
                f.write(relation[idx][4:])
                f.write("\t")
                f.write(str(1))

    def update(self, sentence, e1, e2, e1_position, e2_position, relation, label, cursor = None):
        if cursor is None:
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

            while(os.path.exists("../user/%s/result/label/%d" % (self.name, self.cursor + 1))):
                self.cursor += 1
            self.cursor += 1

            with open("../user/%s/cursor" % self.name, "w") as f:
                f.write(str(self.cursor))

        else:
            with open("../user/%s/result/label/%d" % (self.name, cursor), "w") as f:
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

    def statistics(self):
        stat_dict = dict()
        for rel in self.util.relation_dict.keys():
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

    def visualize_data(self):
        p = figure(plot_width = 750, plot_height = 500)
        try:
            with open("../user/%s/figure/data_vis/vis.pkl" % self.name, "rb") as f:
                data_vis = pickle.load(f)
        except FileNotFoundError:
            return p
        except EOFError:
            return p

        colors = itertools.cycle(palette[8]) 
        circle_list = []
        for rel in sorted(data_vis.keys()):
            if rel == "unlabeled":
                size = 4
            else:
                size = 7

            source = ColumnDataSource(data = dict(x = [elem[0] for elem in data_vis[rel]], y = [elem[1] for elem in data_vis[rel]], cursor = [elem[2] for elem in  data_vis[rel]]))
            circle = p.circle('x', 'y', size = size, color = next(colors), source = source)
            circle_list.append((rel, [circle]))

        p.add_tools(HoverTool(tooltips = [("Cursor", "@cursor")]))
        p.add_tools(TapTool(callback = OpenURL(url = "/@cursor")))

        legend = Legend(items = circle_list)
        p.add_layout(legend, 'right')

        p.legend.click_policy = "hide"

        return p

    def visualize_model(self, cursor):
        try:
            with open("../user/%s/figure/model_vis/att.pkl" % self.name, "rb") as f:
                att_list = pickle.load(f)
            with open("../user/%s/figure/model_vis/output.pkl" % self.name, "rb") as f:
                output_list = pickle.load(f)
        except FileNotFoundError:
            return [], ("", "")
        except EOFError:
            return [], ("", "")

        return att_list[cursor], output_list[cursor]

    def loss_graph(self):
        p = figure(plot_width = 600, plot_height = 200, x_axis_label = 'Epoch', y_axis_label = 'Loss')
        try:
            with open("../user/%s/model/loss" % self.name) as f:
                loss_list = f.read().split("\n")
        except FileNotFoundError:
            return p

        try:
            while(loss_list[-1] == ""):
                loss_list = loss_list[:-1]
        except IndexError:
            return p

        p.circle(list(range(len(loss_list))), loss_list)
        p.line(list(range(len(loss_list))), loss_list, line_width=1)

        return p

    def performance_graph(self):
        p = figure(plot_width = 600, plot_height = 200, x_axis_label = 'Epoch', y_axis_label = 'Performance')
        try:
            with open("../user/%s/model/performance" % self.name) as f:
                performance_list = f.read().split("\n")
        except FileNotFoundError:
            return p

        try: 
            while(performance_list[-1] == ""):
                performance_list = performance_list[:-1]
        except IndexError:
            return p

        P_list = []
        R_list = []
        F1_list = []
        for performance in performance_list:
            P, R, F1 = performance.split("\t")
            P_list.append(P)
            R_list.append(R)
            F1_list.append(F1)

        P_circle = p.circle(list(range(0, len(performance_list) * 5, 5)), P_list, color = "blue")
        P_line = p.line(list(range(0, len(performance_list) * 5, 5)), P_list, line_width = 1, color = "blue")
        R_circle = p.circle(list(range(0, len(performance_list) * 5, 5)), R_list, color = "red")
        R_line = p.line(list(range(0, len(performance_list) * 5, 5)), R_list, line_width = 1, color = "red")
        F1_circle = p.circle(list(range(0, len(performance_list) * 5, 5)), F1_list, color = "purple")
        F1_line = p.line(list(range(0, len(performance_list) * 5, 5)), F1_list, line_width = 1, color = "purple")

        legend = Legend(items=[("Precision", [P_circle, P_line]),
            ("Recall", [R_circle, R_line]),
            ("F1 Score", [F1_circle, F1_line])])
        p.add_layout(legend, 'right')

        p.legend.click_policy="hide"

        return p

    def run_classifier(self):
        if self.pid != -1:
            self.stop_classifier()

        p = subprocess.Popen(["python3 rnn_train.py %s %d %d %d > ../user/%s/model/log.txt 2>&1" % (self.name, self.model_info["word_embedding_dim"], self.model_info["position_embedding_dim"], self.model_info["sentence_embedding_dim"], self.name)], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with open("../user/%s/pid" % self.name, "w") as f:
            f.write(str(p.pid))

        self.pid = p.pid

    def stop_classifier(self):
        return
        print(self.pid)
        if self.pid == -1:
            with open("../user/%s/pid" % self.name) as f:
                self.pid = f.read()
            self.pid = int(self.pid)
            if self.pid == -1:
                return
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        self.pid = -1
        with open("../user/%s/pid" % self.name, "w") as f:
            f.write(str(self.pid))


