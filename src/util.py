"""
Defines several helper/utility methods that are used throughout the project
for doing calculations, converting datatypes between forms, etc.
"""

from math import inf, e, log, sqrt
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
  for i, key in enumerate(sorted(list(dictionary.keys()))):
    hashstring += str(key)
    if not isinstance(dictionary[key], Iterable) and dictionary[key] is not None:
      hashstring += "=" + str(dictionary[key])
    if i < len(dictionary.keys()) - 1:
      hashstring += ","
  return hashstring

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


def kl_divergence(domains, P_data, Q_data, query, log_base=e):
  """
  Calculates kl_divergence between two probability distributions.

  Uses the equation:
    D(KL) = Σ P(X) * log(P(X) / Q(X))
  where the summation is summing over all possible values of X

  By default, the log_base is e to measure information in nats. A log_base
  of 2 would measure information in bits.
  """
  Px_and_Qx = get_Px_and_Qx(domains, P_data, Q_data, query)
  if Px_and_Qx is None:
    return None
  kld = 0
  for Pi_and_Qi in Px_and_Qx:
    P_i = Pi_and_Qi[1][0]
    Q_i = Pi_and_Qi[1][1]
    if P_i == Q_i:
      continue
    if P_i == 0 or Q_i == 0:
      return None
    kld += P_i * log(P_i / Q_i, log_base)
  return kld

def hellinger_dist(domains, P_data, Q_data, query):
  Px_and_Qx = get_Px_and_Qx(domains, P_data, Q_data, query)
  if Px_and_Qx is None:
    return None
  summation = 0
  for Pi_and_Qi in Px_and_Qx:
    # [!] Could probably remove the "query" portion of the tuple...
    pi = Pi_and_Qi[1][0]
    qi = Pi_and_Qi[1][1]
    summation += (sqrt(pi) - sqrt(qi)) ** 2
  return (1/sqrt(2)) * sqrt(summation)

def get_Px_and_Qx(domains, P_data, Q_data, query):
  probs = []
  for query_combo in query.combos(domains):
    new_probs = (query_combo, (query_combo.solve(P_data), query_combo.solve(Q_data)))
    if new_probs[1][0] is None or new_probs[1][1] is None:
      return None
    probs.append(new_probs)
  return probs

def max_key(rng, dictionary):
  """
  Returns the key in the dictionary whose value
  is the maximum of all values in the dictionary.
  In the case of a tie, return one of the max keys
  at random.
  Ex:
    dictionary={"X": 4, "Y": 1, "Z": 4, "W": 2}
      => "X" or "Z" (randomly choice)
  """
  max_val = -inf
  keys = []
  for key, val in dictionary.items():
    if val is None:
      continue
    if val > max_val:
      max_val = val
      keys = [key]
    elif val == max_val:
      keys.append(key)
  return rng.choice(keys) if keys else None

def permutations(dictionary):
  """
  Returns all permutations of variable assignments given a
  dictionary mapping var_names: domain. Result is a list of
  dicts with var_names mapping to assignments.
  """
  num_combos = num_permutations(dictionary.values())
  combos = [{} for _ in range(num_combos)]
  for var, domain in sorted(dictionary.items()):
    num_combos /= len(domain)
    count = 0
    pos_val_index = 0
    for combo in combos:
      if count >= num_combos:
        count = 0
        pos_val_index = pos_val_index + 1 if pos_val_index + 1 < len(domain) else 0
      combo[var] = domain[pos_val_index]
      count += 1
  return combos

def num_permutations(list_of_lists):
  """
  Returns the number of permutations from an input
  list of lists, representing domains.
  """
  if not list_of_lists:
    return 0
  count = 1
  for lst in list_of_lists:
    count *= len(lst)
  return count

def only_dicts_with_givens(dicts, assignments={}):
  """
  Takes a list of dictionaries, and returns a subset
  list only containing dictionaries whose key:value pairs
  are consistent with the parameterized assignments.

  Ex:
  dicts=[{"A": 1, "B": 0}, {"A": 0, "B": 1}], assignments={"A": 0}
    => [{"A": 0, "B": 1}]
  """
  if not assignments:
    return dicts
  res = dicts.copy()
  for d in dicts:
    for a in assignments:
      if d[a] != assignments[a]:
        res.remove(d)
        break
  return res

def only_given_keys(dictionary, keys):
  """
  Outputs a dictionary with the key:values of an
  original dictionary, but only with items whose
  keys are specified as a parameter.
  """
  res = dict(dictionary)
  for key in dictionary:
    if key not in keys:
      del res[key]
  return res

class Counter(dict):
  def __getitem__(self, idx):
    self.setdefault(idx, 0)
    return dict.__getitem__(self, idx)
  
  def copy(self):
    return Counter(dict.copy(self))

def printProgressBar (iteration, total, prefix = '', suffix = '', length = 50, fill = '█', printEnd = "\r"):
    """
    Author: greenstick (Stack Overflow/GitHub)
    Call in a loop to create terminal progress bar
    Parameters:
      iteration   - Required  : current iteration (Int)
      total       - Required  : total iterations (Int)
      prefix      - Optional  : prefix string (Str)
      suffix      - Optional  : suffix string (Str)
      decimals    - Optional  : positive number of decimals in percent complete (Int)
      length      - Optional  : character length of bar (Int)
      fill        - Optional  : bar fill character (Str)
      printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = "%04.1f" % (100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    space = ' ' if float(percent) < 100 else ''
    print(f'\r{space}{percent}% |{bar}| {suffix}', end = printEnd)
