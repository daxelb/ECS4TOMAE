import unittest
import util

class Tests(unittest.TestCase):
  def test_kl_divergence(self):
    domain = {"X": (0,1,2)}
    P_data = {"X": []}
    Q_data = {"X": []}
    [P_data["X"].append(0) for _ in range(36)]
    [P_data["X"].append(1) for _ in range(48)]
    [P_data["X"].append(2) for _ in range(16)]
    [Q_data["X"].extend([0,1,2]) for _ in range(33)]
    query = util.parse_query("P(X)")[0]
    kl_div = util.kl_divergence(domain, P_data, Q_data, query[0], query[1])
    rounded = round(kl_div, 8)
    self.assertEqual(rounded, 0.0852996)

if __name__ == "__main__":
  unittest.main()