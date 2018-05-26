import time

while(True):
    with open("temp.txt", "a") as f:
        f.write("HI\n")
    time.sleep(5)
