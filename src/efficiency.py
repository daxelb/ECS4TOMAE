import time
import pandas as pd
import numpy as np


startTime = time.time()
row_list = []
for _ in range(300):
  row_list.append(dict((var, np.random.randint(2)) for var in ["X", "Y", "Z", "W"]))
  count = 0
  total = 0
  for row in row_list:
    if row["W"] == 1:
      total += 1
      if row["X"] == 0:
        count += 1
print(time.time() - startTime)
        
startTime = time.time()
rows = pd.DataFrame(index=np.arange(1000), columns=sorted(list({"X", "Y", "Z", "W"})))
for _ in range(300):
  rows = rows.append(dict((var, np.random.randint(2)) for var in ["X", "Y", "Z", "W"]), ignore_index=True)
  query_space = rows.query("W == 1")
  query_space.query("X == 0")
print(time.time() - startTime)

startTime = time.time()
rows = pd.DataFrame(columns=sorted(list({"X", "Y", "Z", "W"})))
for _ in range(300):
  rows = rows.append(dict((var, np.random.randint(2)) for var in ["X", "Y", "Z", "W"]), ignore_index=True)
  query_space = rows.query("W == 1")
  query_space.query("X == 0")
print(time.time() - startTime)