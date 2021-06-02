import math
import numpy as np
import random

def max_key(dictionary):
  return max(dictionary, key=dictionary.get)

def dict_of_lists_to_list_of_dicts(dictionary):
  res = [{}] * len(list(dictionary.values())[0])
  for key, value in dictionary.items():
    for i, e in enumerate(value):
      res[i][key] = e
  return res

# def subtract_lists(list1, list2):
#   return [e for e in list1 if e not in list2]

def random_assignment(domains):
  return random.choice(permutations(domains))

def get_nodes_from_edges(obs_edges, latent_edges=None):
  edges = obs_edges
  if latent_edges:
    edges.extend(e for e in latent_edges if e not in edges)
  nodes = list()
  for tup in edges:
    for i in range(len(tup)):
      if tup[i] not in nodes:
        nodes.append(tup[i])
  return nodes

def num_permutations(list_of_lists):
  """
  Returns the number of permutations from an input
  list of lists, representing domains.
  """
  if not len(list_of_lists):
    return 0
  count = 1
  for lst in list_of_lists:
    count *= len(lst)
  return count

def permutations(dictionary):
  """
  Returns all permutations of variable assignments given a
  dictionary mapping var_names: domain. Result is a list of
  dicts with var_names mapping to assignments.
  """
  num_combos = num_permutations(dictionary.values())
  combos = [{} for _ in range(num_combos)]
  for item in dictionary.items():
    var = item[0]
    domain = item[1]
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

def hash_from_dict(dictionary):
  hashable = ""
  for i, key in enumerate(dictionary.keys()):
    hashable += str(key) + "=" + str(dictionary[key])
    if i < len(dictionary.keys()) - 1:
      hashable += ","
  return hashable

def hashes_from_domain(domain_dict):
  return [hash_from_dict(d) for d in permutations(domain_dict)]

def dict_from_hash(hash_string):
  res = dict()
  key_bool = True
  key = ""
  val = ""
  for char in hash_string:
    if char == '=':
      key_bool = False
    elif char == ',':
      key_bool = True
      res[key] = int(float(val))
      key = ""
      val = ""
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
  for sample in only_dicts_with_givens(dataset, givens):
    key = hash_from_dict(
        only_specified_keys(sample, action_vars))
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
    expected_vals[key] = avg(dictionary[key])
  return expected_vals

def expected_vals(dataset, action_vars, reward_var, givens={}):
  """
  Returns a dictionary mapping action_vars to their
  expected/mean reward value using two previously defined
  methods.
  """
  return expected_vals_from_rewards(reward_vals(dataset, action_vars, reward_var, givens))

def avg(lst):
  """
  Returns the average/mean/expected value of a list
  of numbers
  """
  return sum(lst) / len(lst)

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
  res = 0
  for i in range(len(Px)):
    if Px[i][0] != Qx[i][0]:
      print("Assignments of P and Q are unmated in kl_divergence() method")
      return
    res += Px[i][1] * math.log(Px[i][1] / Qx[i][1], log_base)
  return res

def only_specified_keys(dictionary, keys):
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
  res = list(dicts)
  for d in dicts:
    for a in assignments:
      if d[a] != assignments[a]:
        res.remove(d)
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
  queries = permutations(domains)
  unused_keys = queries[0].keys() - Q.keys()
  for q in permutations(domains):
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
    new_e = only_specified_keys(q, e.keys())
    assignment = only_specified_keys(q, [q for q in Q_and_e if Q_and_e[q] == None])
    probs.append((assignment, uncond_prob(dataset, q) / uncond_prob(dataset, new_e)))
  return probs

def prob(dataset, Q, e={}):
  """
  For a query for which all queried-on variables
  are assigned, returns the conditional probability 
  calculated from the dataset.
  """
  if has_unassigned(Q) or has_unassigned(Q):
    print("All variables should have assignments to use util.prob(). Try util.prob_with_unassigned(), instead.")
  return uncond_prob(dataset, {**Q, **e}) / uncond_prob(dataset, e)

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
  total = len(dataset)
  count = 0
  for i in range(total):
    consistent = True
    for key in Q:
      if Q[key] != None and dataset[i][key] != Q[key]:
        consistent = False
        break
    count += consistent
  return count / total

def split_query(str_query):
  """
  Helper method that splits a probability distribution
  into its single parts.

  Ex:
  "P(X)P(X|Y)P(X|Y,W)"
    => ["P(X)", "P(X|Y)", "P(X|Y,W)"]
  """
  return ["P"+e for e in str_query.split("P") if e]

def parse_query(str_query):
  """
  Parses a (tier-1) probability query into a form
  that is meaningful and the program can parse.

  Ex:
  P(X=1)P(Y|X=1)
    => [({"X": 1}, {}), ({"Y": None}, {"X": 1})]
  """
  parsed = []
  for q in split_query(str_query):
    query = True
    assignment = False
    Q_var = Q_val = e_var = e_val = ""
    new_p = [{}, {}]
    for char in q:
      if char == '(':
        Q_var = ""
      elif char == '|':
        assignment = False
        new_p[0][Q_var] = None if Q_val == "" else float(Q_val) if int(
            float(Q_val)) != float(Q_val) else int(float(Q_val))
        query = False
      elif char == ')':
        if query:
          new_p[0][Q_var] = None if Q_val == "" else float(Q_val) if int(
              float(Q_val)) != float(Q_val) else int(float(Q_val))
        elif len(e_var):
          new_p[1][e_var] = None if e_val == "" else float(e_val) if int(
              float(e_val)) != float(e_val) else int(float(e_val))
        parsed.append(tuple(new_p))
        Q_var = Q_val = e_var = e_val = ""
        new_p = [{}, {}]
      elif char == ',':
        assignment = False
        if query:
          new_p[0][Q_var] = None if Q_val == "" else float(Q_val) if int(
              float(Q_val)) != float(Q_val) else int(float(Q_val))
          Q_var = Q_val = e_var = e_val = ""
        else:
          new_p[1][e_var] = None if e_val == "" else float(e_val) if int(
              float(e_val)) != float(e_val) else int(float(e_val))
          Q_var = Q_val = e_var = e_val = ""
      elif char == '=':
        assignment = True
      else:
        if query:
          if assignment:
            Q_val += char
          else:
            Q_var += char
        else:
          if assignment:
            e_val += char
          else:
            e_var += char
  return parsed

if __name__ == "__main__":
  d = {'X': np.array([1, 1, 1, 1, 1]), 'Z': np.array([1, 1, 1, 1, 1]),
       'W': np.array([1, 0, 0, 1, 0]), 'Y': np.array([1, 0, 0, 1, 0])}
  print(dict_of_lists_to_list_of_dicts(d))
