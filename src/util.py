def alphabetize(l):
  l.sort()
  return l

def num_combinations(list_of_lists):
  if not len(list_of_lists):
    return 0
  count = 1
  for lst in list_of_lists:
    count *= len(lst)
  return count

def combinations(dictionary):
  combos = list()
  num_combos = num_combinations(dictionary.values())

  for _ in range(num_combos):
    combos.append(dict())
    # combos.append([None] * len(dictionary.keys()))
  # index was previously in here w/ enumerate()
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
      # combo[index] = (var, domain[pos_val_index])
      count += 1
  return combos

def only_specified_keys(dictionary, keys):
  res = dict(dictionary)
  for key in dictionary:
    if key not in keys:
      del res[key]
  return res

def query_combos(domains, Q):
  queries = combinations(domains)
  unused_keys = queries[0].keys() - Q.keys()
  for q in combinations(domains):
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

def exists_unassigned(Q):
  for q in Q:
    if Q[q] == None:
      return True
  return False

def prob_with_unassigned(dataset, domains, Q, e={}):
  if not exists_unassigned(Q) and not exists_unassigned(e):
    return prob(dataset, Q, e)
  probs = list()
  Q_and_e = {**Q, **e}
  for q in query_combos(domains, Q_and_e):
    new_e = only_specified_keys(q, e.keys())
    assignment = only_specified_keys(q, [q for q in Q_and_e if Q_and_e[q] == None])
    probs.append((assignment, uncond_prob(dataset, q) / uncond_prob(dataset, new_e)))
  return probs

def prob(dataset, Q, e={}):
  if exists_unassigned(Q) or exists_unassigned(Q):
    print("All variables should have assignments to use util.prob(). Try util.prob_with_unassigned(), instead.")
  return uncond_prob(dataset, {**Q, **e}) / uncond_prob(dataset, e)

def uncond_prob(dataset, Q):
  if not Q:
    return 1.0
  total = len(list(dataset.values())[0])
  count = 0
  for i in range(len(list(dataset.values())[0])):
    consistent = True
    for key in Q:
      if Q[key] != None and dataset[key][i] != Q[key]:
        consistent = False
        break
    count += consistent
  return count / total

def split_query(str_query):
  return ["P"+e for e in str_query.split("P") if e]

def parse_query(str_query):
  parsed = []
  for q in split_query(str_query):
    query = True
    assignment = False
    Q_var = Q_val = e_var = e_val = ""
    new_p = [{}, {}]
    for char in q:
      if char == 'P' or char == '(':
        query = True
        assignment = False
      elif char == '|':
        assignment = False
        Q_val = None if Q_val == "" else float(Q_val) if int(float(Q_val)) != float(Q_val) else int(float(Q_val))
        new_p[0][Q_var] = Q_val
        query = False
      elif char == ')':
        if query:
          Q_val = None if Q_val == "" else float(Q_val) if int(float(Q_val)) != float(Q_val) else int(float(Q_val))
          new_p[0][Q_var] = Q_val
        elif len(e_var):
          e_val = None if e_val == "" else float(e_val) if int(float(e_val)) != float(e_val) else int(float(e_val))
          new_p[1][e_var] = e_val
        parsed.append(tuple(new_p))
        Q_var = Q_val = e_var = e_val = ""
        new_p = [{}, {}]
      elif char == ',':
        assignment = False
        if query:
          Q_val = None if Q_val == "" else float(Q_val) if int(float(Q_val)) != float(Q_val) else int(float(Q_val))
          new_p[0][Q_var] = Q_val
          Q_var = Q_val = e_var = e_val = ""
        else:
          e_val = None if e_val == "" else float(e_val) if int(float(e_val)) != float(e_val) else int(float(e_val))
          new_p[1][e_var] = e_val
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