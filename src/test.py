import time
from numpy import random

def func():
  rnge = 9999999
  start1 = time.time()
  for _ in range(rnge):
    if random.rand() < 0.872387167328761614786:
      continue
    else:
      continue
  end1 = time.time()
  start2 = time.time()
  for _ in range(rnge):
    if random.rand() < 0.8724:
      continue
    else:
      continue
  end2 = time.time()
  start3 = time.time()
  one = 1.0
  for _ in range(rnge):
    if one == 1.0 or random.rand() < one:
      continue
    else:
      continue
  end3 = time.time()
  print(end1-start1, end2-start2, end3-start3)
  
func()