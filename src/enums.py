from enum import Enum

class Policy(Enum):
  ADJUST = "Adjust"
  NAIVE = "Naive"
  SENSITIVE = "Sensitive"
  SOLO = "Solo"

  def __lt__(self, other):
    if self.__class__ is other.__class__:
      return self.value < other.value
    return NotImplemented  

class ASR(Enum):
  G = "Greedy"
  EG = "Epsilon Greedy"
  EF = "Epsilon First"
  ED = "Epsilon Decreasing"
  TS = "Thompson Sampling"

  def __lt__(self, other):
    if self.__class__ is other.__class__:
      return self.value < other.value
    return NotImplemented  