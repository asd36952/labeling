import os

class user():
    def __init__(self):
        self.is_active = False
        self.name = "Unknown"

    def login(self, name, password):
        if name not in os.listdir("../user/"):
            return 0, "Wrong User Name."
        with open("../user/%s" % name) as f:
            if f.read()[:-1] == password:
                self.is_active = True
                self.name = name
                with open("../user/%s.cursor" % name) as f2:
                    self.cursor = int(f2.read()[:-1])
                return 1, "Success."
        return 0, "Wrong Password."

    def register(self, name, password):
        if name in os.listdir("../user/"):
            return 0, "The User Name already exist."
        with open("../user/%s" % name, "w") as f:
            f.write(password)
        with open("../user/%s.cursor" % name, "w") as f:
            f.write("0")
        return 1, "Success."
