import time
from numpy import random
import pandas as pd
import numpy as np

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
  
num_eps = 50000000
n = 10000

start = time.time()
a = [None] * num_eps
for i in range(num_eps):
  a[i] = a[i-1] + random.rand() if i > 0  else 0
print(time.time() - start)

start = time.time()
a = [0] * num_eps
for i in range(num_eps):
  if i > 0:
    a[i] = a[i-1] + random.rand()
print(time.time() - start)

start = time.time()
a = [0]
for _ in range(num_eps - 1):
  a.append(a[-1] + random.rand())
print(time.time() - start)