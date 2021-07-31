from enum import Enum

class Datatype(Enum):
  OBS = "obs"
  EXP = "exp"
  
class Policy(Enum):
  DEAF = "Deaf"
  NAIVE = "Naive"
  SENSITIVE = "Sensitive"
  ADJUST = "Adjust"
  
class IV(Enum):
  POL = "Policy"
  EPS = "Epsilon"
  DNC = "Divergent Node Confidence"
  SN = "Samples Needed"