import math
import gutil
from collections.abc import Iterable

def hash_from_dict(dictionary):
  """
  Creates a hashable string from a dictionary
  that maps values to their assignments.
  Ex:
  dictionary={"A": 1, "B": 5, "C": 1}
    => "A=1,B=5,C=1"
  """
  hashstring = ""
  for i, key in enumerate(dictionary.keys()):
    hashstring += str(key)
    if not isinstance(dictionary[key], Iterable):
      hashstring += "=" + str(dictionary[key])
    if i < len(dictionary.keys()) - 1:
      hashstring += ","
  return hashstring

def hashes_from_domain(domain_dict):
  """
  Creates a list of hashstrings representing variable
  assignments from their possible values in a domain.
  Ex:
  domain_dict={"A": [0,1], "B": [0,1,2]}
    => ["A=0,B=0", "A=0,B=1", "A=0,B=2", "A=1,B=0", "A=1,B=1", "A=1,B=2"]
  """
  return [hash_from_dict(d) for d in gutil.permutations(domain_dict)]

def dict_from_hash(hashstring):
  """
  From a hashstring (like one created from hash_from_dict),
  creates and returns a dictionary mapping variables to their
  assignments.
  Ex:
  hashstring="A=0,B=1"
    => {"A": 0, "B": 1}
  """
  if hashstring is None:
    return None
  if len(hashstring) == 0:
    return {}
  res = dict()
  key_bool = True
  key = val = ""
  for char in hashstring:
    if char == '=':
      key_bool = False
    elif char == ',':
      key_bool = True
      res[key] = int(float(val))
      key = val = ""
    else:
      if key_bool:
        key += char
      else:
        val += char
  if len(val):
    res[key] = int(float(val))
  return res

def reward_vals(dataset, action_vars, reward_var, givens={}):
  """
  From a dataset, returns the rewards associated with different
  values of action_variables. Formatted as a dictionary mapping
  action_var to a list of rewards.
  """
  reward_vals = dict()
  for sample in gutil.only_dicts_with_givens(dataset, givens):
    key = hash_from_dict(gutil.only_given_keys(sample, action_vars))
    if key not in reward_vals:
      reward_vals[key] = list()
    reward_vals[key].append(sample[reward_var])
  return reward_vals

def expected_vals_from_rewards(dictionary):
  """
  Calculates the expected value of keys in a dictionary
  whose parameterized value is a list of numbers.
  Result is a dictionary mapping the keys of the entry
  dictionary to the average value of the key (from a list).

  Although the method is named to take "rewards" as the
  param, it could be any dictionary whose keys map to lists of
  numbers.
  """
  expected_vals = {}
  for key in dictionary:
    expected_vals[key] = gutil.avg(dictionary[key])
  return expected_vals

def expected_vals(dataset, action_vars, reward_var, givens={}):
  """
  Returns a dictionary mapping action_vars to their
  expected/mean reward value using two previously defined
  methods.
  """
  return expected_vals_from_rewards(reward_vals(dataset, action_vars, reward_var, givens))

def kl_divergence(domains, P_data, Q_data, query, log_base=math.e):
  """
  Calculates kl_divergence between two probability distributions.

  Uses the equation:
    D(KL) = Σ P(X) * log(P(X) / Q(X))
  where the summation is summing over all possible values of X

  By default, the log_base is e to measure information in nats. A log_base
  of 2 would measure information in bits.
  """
  Px = prob_with_unassigned(domains, P_data, query)
  Qx = prob_with_unassigned(domains, Q_data, query)
  if Px is None or Qx is None:
    return None
  res = 0
  for i in range(len(Px)):
    assert Px[i][0] == Qx[i][0]
    res +=  0 if Qx[i][1] == Px[i][1] else \
            math.inf if Qx[i][1] == 0 else \
            -math.inf if Px[i][1] == 0 else \
            Px[i][1] * math.log(Px[i][1] / Qx[i][1], log_base)
  return res

def has_unassigned(Q):
  """
  Returns True if the input query contains
  an unassigned variable, else False
  """
  for q in Q:
    if Q[q] == None:
      return True
  return False

def prob_with_unassigned(domains, dataset, query):
  probs = []
  for query_combo in query.combos(domains):
    new_prob = (query_combo, query_combo.solve(dataset))
    if new_prob[1] is None:
      return None
    probs.append(new_prob)
  return probs