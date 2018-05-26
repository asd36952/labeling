import subprocess
import time

a = subprocess.Popen(["nohup python3 temp.py"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
for i in range(100):
    time.sleep(1)
    print(i)
    print(a)
    with open("temp.txt") as f:
        print(f.read())
