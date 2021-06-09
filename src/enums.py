from enum import Enum

class Datatype(Enum):
  OBS = "obs"
  EXP = "exp"
  
class Policy(Enum):
  DEAF = 0
  NAIVE = 1
  SENSITIVE = 2
  TRANSPORT = 3
  
class Result(Enum):
  CUM_REGRET = "Cumulative Regret"
  PERC_CORR = "% Correct Divergence IDs"