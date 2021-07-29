from enum import Enum

class Datatype(Enum):
  OBS = "obs"
  EXP = "exp"
  
class Policy(Enum):
  DEAF = "Deaf"
  NAIVE = "Naive"
  SENSITIVE = "Sensitive"
  ADJUST = "Adjust"
  
class Result(Enum):
  CUM_REGRET = "Pseudo Cumulative Regret"
  PERC_CORR = "% Correct Divergence IDs"