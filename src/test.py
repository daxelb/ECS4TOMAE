import unittest
import util

class Tests(unittest.TestCase):
  def test_kl_divergence(self):
    domain = {"X": (0,1,2)}
    P_data = []
    Q_data = []
    [P_data.append({"X": 0}) for _ in range(36)]
    [P_data.append({"X": 1}) for _ in range(48)]
    [P_data.append({"X": 2}) for _ in range(16)]
    [Q_data.extend([{"X": 0}, {"X": 1}, {"X": 2}]) for _ in range(33)]
    query = util.parse_query("P(X)")[0]
    self.assertEqual(round(util.kl_divergence(
        domain, P_data, Q_data, query[0], query[1]), 8), 0.0852996)

  def test_hash_from_dict(self):
    self.assertEqual(util.hash_from_dict(
        {"X": 0, "W": 1, "Z": 123}), "X=0,W=1,Z=123")
  
  def test_dict_from_hash(self):
    self.assertEqual(util.dict_from_hash(
        "X=0,W=1,Z=123"), {"X": 0, "W": 1, "Z": 123})

if __name__ == "__main__":
  unittest.main()
