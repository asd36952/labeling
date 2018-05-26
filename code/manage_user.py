import os

class user():
    def __init__(self):
        self.is_active = False
        self.name = "Unknown"

    def login(self, name, password):
        if name not in os.listdir("../user/"):
            return 0, "Wrong User Name."
        with open("../user/%s/info" % name) as f:
            if f.read()[:-1] == password:
                self.is_active = True
                self.name = name
                with open("../user/%s/cursor" % name) as f2:
                    self.cursor = int(f2.read())
                return 1, "Success."
        return 0, "Wrong Password."

    def register(self, name, password):
        if name in os.listdir("../user/"):
            return 0, "The User Name already exist."
        os.makedirs("../user/%s/" % name)
        with open("../user/%s/info" % name, "w") as f:
            f.write(password)
        with open("../user/%s/cursor" % name, "w") as f:
            f.write("0")
        with open("../user/%s/pid" % name, "w") as f:
            f.write("-1")
        os.makedirs("../user/%s/model" % name)
        os.makedirs("../user/%s/figure" % name)
        os.makedirs("../user/%s/figure/data_vis" % name)
        os.makedirs("../user/%s/figure/model_vis" % name)
        os.makedirs("../user/%s/result" % name)
        os.makedirs("../user/%s/result/label/" % name)
        os.makedirs("../user/%s/result/train/" % name)
        os.makedirs("../user/%s/result/valid/" % name)
        return 1, "Success."

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
            f.write("\n")

        self.cursor += 1
        with open("../user/%s/cursor" % self.name, "w") as f:
            f.write(str(self.cursor))

    def statistics(self, util):
        stat_dict = dict()
        for rel in util.relation_dict.keys():
            stat_dict[rel] = [0, 0]
        stat_dict["no_relation"] = [0, 0]
        for data_name in os.listdir("../user/%s/result/label/" % self.name):
            with open("../user/%s/result/label/%s" % (self.name, data_name)) as f:
                line = f.read()
            while(line[-1] == "\n"):
                line = line[:-1]
            sent, e1, e2, e1_position, e2_position, rel, label = line.split("\t")
            label = int(label)

            stat_dict[rel][label] += 1

        return stat_dict
