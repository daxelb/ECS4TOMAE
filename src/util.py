import math
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
    if not isinstance(dictionary[key], Iterable):
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


def kl_divergence(domains, P_data, Q_data, query, log_base=math.e):
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
  for i in range(len(Px_and_Qx)):
    P_i = Px_and_Qx[i][1][0]
    Q_i = Px_and_Qx[i][1][1]
    kld +=  0 if P_i == Q_i else \
            math.inf if Q_i == 0 else \
            -math.inf if P_i == 0 else \
            P_i * math.log(P_i / Q_i, log_base)
  return kld

def get_Px_and_Qx(domains, P_data, Q_data, query):
  probs = []
  for query_combo in query.combos(domains):
    new_probs = (query_combo, (query_combo.solve(P_data), query_combo.solve(Q_data)))
    if new_probs[1][0] is None or new_probs[1][1] is None:
      return None
    probs.append(new_probs)
  return probs