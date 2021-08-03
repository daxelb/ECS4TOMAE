from enum import Enum
class Policy(Enum):
  SOLO = "Solo"
  NAIVE = "Naive"
  SENSITIVE = "Sensitive"
  ADJUST = "Adjust"
  
class ASR(Enum):
  GREEDY = "Greedy"
  EPSILON_GREEDY = "Epsilon Greedy"
  EPSILON_FIRST = "Epsilon First"
  EPSILON_DECREASING = "Epsilon Decreasing"
  THOMPSON_SAMPLING = "Thompson Sampling"