import time
from numpy import random
import pandas as pd

def get_add_pd(num_episodes):
  dat = [random.rand() for _ in range(num_episodes)]
  return pd.DataFrame(data=[dat], columns=range(num_episodes))

def res_pd(num_episodes, N):
  res = pd.DataFrame(columns=range(num_episodes))
  for _ in range(N):
    res = res.append(get_add_pd(num_episodes))
  return res.mean()

def get_add_list(num_episodes):
  return [random.rand() for _ in range(num_episodes)]

def res_list(num_episodes, N):
  return [get_add_list(num_episodes) for _ in range(N)]
  
num_eps = 400
n = 10000
# start = time.time()
# res_pd(num_eps, n)
# print(time.time() - start)
# start = time.time()
# res_list(num_eps, n)
# print(time.time() - start)

start = time.time()
a = {}
for _ in range(n):
  data = random.rand()
  key = "a" if data < 0.5 else "b" if data > 0.75 else "c"
  if key not in a:
    a[key] = []
  a[key].append(get_add_list(num_eps))
print(time.time() - start)

start = time.time()
a = {}
for _ in range(n):
  data = random.rand()
  key = "a" if data < 0.5 else "b" if data > 0.75 else "c"
  if key not in a:
    a[key] = [get_add_list(num_eps)]
    continue
  a[key].append(get_add_list(num_eps))
print(time.time() - start)
  
