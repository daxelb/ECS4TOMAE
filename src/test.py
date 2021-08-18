import time
from numpy import random
import pandas as pd
import numpy as np
import math

def get_add_pd(T):
  dat = [random.rand() for _ in range(T)]
  return pd.DataFrame(data=[dat], columns=range(T))

def res_pd(T, N):
  res = pd.DataFrame(columns=range(T))
  for _ in range(N):
    res = res.append(get_add_pd(T))
  return res.mean()

def get_add_list(T):
  return [random.rand() for _ in range(T)]

def res_list(T, N):
  return [get_add_list(T) for _ in range(N)]
  
# num_eps = 50000000
# n = 10000

# start = time.time()
# a = [None] * num_eps
# for i in range(num_eps):
#   a[i] = a[i-1] + random.rand() if i > 0  else 0
# print(time.time() - start)

# start = time.time()
# a = [0] * num_eps
# for i in range(num_eps):
#   if i > 0:
#     a[i] = a[i-1] + random.rand()
# print(time.time() - start)

# start = time.time()
# a = [0]
# for _ in range(num_eps - 1):
#   a.append(a[-1] + random.rand())
# print(time.time() - start)

n = 1000000
# start = time.time()
rng = random.default_rng()
# for _ in range(n):
#   a = rng.random()
# print(time.time() - start)

# start = time.time()
# for _ in range(n):
#   a = random.rand()
# print(time.time() - start)

# start = time.time()
# best = []
# for i in range(n):
#   choices = {"a": int(rng.random()), "b": int(rng.random()), "c": int(rng.random())}
#   best_choices = []
#   best_rew = float('-inf')
#   for key, value in choices.items():
#     if value > best_rew:
#       best_choices = [key]
#       best_rew = value
#     elif value == best_rew:
#       best_choices.append(key)
#   best.append(rng.choice(best_choices) if best_choices else None)
# print(time.time() - start)

# start = time.time()
# best = []
# for i in range(n):
#   choices = {"a": int(rng.random()), "b": int(rng.random()), "c": int(rng.random())}
#   best_choices = None
#   best_rew = float('-inf')
#   for key, value in choices.items():
#     if value > best_rew:
#       best_choice = key
#   best.append(best_choice)
# print(time.time() - start)

start = time.time()
best = []
for i in range(n):
  choices = {"a": int(rng.random()), "b": int(rng.random()), "c": int(rng.random())}
  best_choices = None
  best_rew = -999
  for key, value in choices.items():
    if value > best_rew:
      best_choice = key
  best.append(best_choice)
print(time.time() - start)

start = time.time()
best = []
for i in range(n):
  choices = {"a": int(rng.random()), "b": int(rng.random()), "c": int(rng.random())}
  best_choices = None
  best_rew = -math.inf
  for key, value in choices.items():
    if value > best_rew:
      best_choice = key
  best.append(best_choice)
print(time.time() - start)