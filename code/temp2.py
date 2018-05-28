import subprocess
import time
import os
import signal

a = subprocess.Popen(["nohup python3 temp.py"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
for i in range(10):
    time.sleep(1)
    print(i)
    print(a.pid)
    with open("temp.txt") as f:
        print(f.read())
    print()
os.kill(a.pid, signal.SIGTERM)
