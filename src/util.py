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

def kl_divergence(domains, P_data, Q_data, query, e, log_base=math.e):
  """
  Calculates kl_divergence between two probability distributions.

  Uses the equation:
    D(KL) = Σ P(X) * log(P(X) / Q(X))
  where the summation is summing over all possible values of X

  By default, the log_base is e to measure information in nats. A log_base
  of 2 would measure information in bits.
  """
  Px = prob_with_unassigned(domains, P_data, query, e)
  Qx = prob_with_unassigned(domains, Q_data, query, e)
  if Px is None or Qx is None:
    return None
  res = 0
  for i in range(len(Px)):
    assert Px[i][0] == Qx[i][0]
    res += 0 if Qx[i][1] == Px[i][1] else \
        math.inf if Qx[i][1] == 0 else \
        -math.inf if Px[i][1] == 0 else \
        Px[i][1] * math.log(Px[i][1] / Qx[i][1], log_base)
  return res

def query_combos(domains, Q):
  """
  Used when a query contains unassigned/unspecified variables.
  Returns a list of fully-specified queries where previously
  unassigned variables are given assignments from their domain.

  Ex:
  P(Y|X=1) where domain of Y = {0,1}
    => [ P(Y=0|X=1), P(Y=1|X=1) ]
  """
  queries = gutil.permutations(domains)
  unused_keys = queries[0].keys() - Q.keys()
  for q in gutil.permutations(domains):
    for var in Q:
      if q in queries and Q[var] != None and q[var] != Q[var]:
        queries.remove(q)
  queries_no_dupes = []
  for q in queries:
    for key in unused_keys:
      del q[key]
    if q not in queries_no_dupes:
      queries_no_dupes.append(q)
  return queries_no_dupes

def has_unassigned(Q):
  """
  Returns True if the input query contains
  an unassigned variable, else False
  """
  for q in Q:
    if Q[q] == None:
      return True
  return False

def prob_with_unassigned(domains, dataset, Q, e={}):
  """
  For unassigned variables, calculates the conditional
  probability query for all possible assignment values.

  Returns a list of tuples where the first element is
  the assigned values of the unspecified variables (as
  a dictionary) and the second element is the calculated
  probability.
  """
  if not has_unassigned(Q) and not has_unassigned(e):
    return prob(dataset, Q, e)
  probs = list()
  Q_and_e = {**Q, **e}
  for q in query_combos(domains, Q_and_e):
    new_e = gutil.only_given_keys(q, e.keys())
    assignment = gutil.only_given_keys(q, [q for q in Q_and_e if Q_and_e[q] == None])
    prob_new_e = uncond_prob(dataset, new_e)
    if not prob_new_e:
      return None
    probs.append((assignment, uncond_prob(dataset, q) / prob_new_e))
  return probs

def prob(dataset, Q, e={}):
  """
  For a query for which all queried-on variables
  are assigned, returns the conditional probability 
  calculated from the dataset.
  """
  assert not (has_unassigned(Q) or has_unassigned(e))
  prob_e = uncond_prob(dataset, e)
  if prob_e == 0:
    return None
  return uncond_prob(dataset, {**Q, **e}) / prob_e

def consistent(dataset, Q):
  """
  Calculates the number of datapoints in the dataset
  that are consistent with a probability query.
  """
  count = 0
  for datapoint in dataset:
    count += all([Q[key] == datapoint[key] for key in Q])
  return count

def uncond_prob(dataset, Q):
  """
  Calculates an probability query that excludes
  evidence/conditional arguments that would follow
  a conditioning bar.
  
  Ex:
  P(A,B) = P(A ∩ B)
    => 0.5
  """
  if not Q:
    return 1.0
  return consistent(dataset, Q) / len(dataset)