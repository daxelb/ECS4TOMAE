import numpy as np

random_seed = int(np.random.rand() * ((2**32) - 1))
# print(random_seed)
np.random.seed(555)
print([np.random.rand() for _ in range(1000)][99])
import numpy as np

print([np.random.rand() for _ in range(1000)][99])
