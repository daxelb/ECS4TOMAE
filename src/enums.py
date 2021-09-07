from enum import Enum
class Policy(Enum):
    ADJUST = "Adjust"
    NAIVE = "Naive"
    SENSITIVE = "Sensitive"
    SOLO = "Solo"

class ASR(Enum):
    G = "Greedy"
    EG = "Epsilon Greedy"
    EF = "Epsilon First"
    ED = "Epsilon Decreasing"
    TS = "Thompson Sampling"