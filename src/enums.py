"""
Defines OTP (Observational Transport Policy) and ASR (Action Selection Rule)
Enums used throughout this project
"""

from enum import Enum


class OTP(Enum):
  ADJUST = "Adjust"
  NAIVE = "Naive"
  SENSITIVE = "Sensitive"
  SOLO = "Solo"

  def __lt__(self, other):
    if self.__class__ is other.__class__:
      return self.value < other.value
    return NotImplemented

  def __str__(self):
    return self.value


class ASR(Enum):
  EG = 'EG'  # "Epsilon Greedy"
  EF = 'EF'  # "Epsilon First"
  ED = 'ED'  # "Epsilon Decreasing"
  TS = 'TS'  # "Thompson Sampling"

  def __lt__(self, other):
    if self.__class__ is other.__class__:
      return self.value < other.value
    return NotImplemented

  def __str__(self):
    return self.value

  def __repr__(self):
    return str(self)


if __name__ == '__main__':
  a = ASR.EG
  b = ASR.EF
  c = ASR.ED
  d = ASR.TS
  print(hash(a), hash(b), hash(c), hash(d))
