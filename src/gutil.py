import math
import random

def avg_list(lists):
  """
  Returns averaged values of elements in list of lists
  Ex:
    lists = [[0,1,2],[5,2,1],[4,3,0]]
      => [(0+5+4)/3, (1+2+3)/3, (2+1+0)/3]
      => [3,2,1]
  """
  averaged_list = [0] * len(lists[0])
  for lst in lists:
    for i, e in enumerate(lst):
      averaged_list[i] += e / len(lists)
  return averaged_list

def first_value(dictionary):
  """
  Returns the value of the first entry in a dictionary.
  """
  return list(dictionary.values())[0]

def first_key(dictionary):
  """
  Returns the key of the first entry in a dictionary.
  """
  return list(dictionary.keys())[0]

def remove_dupes(lst):
  """
  Removes duplicates from an input list and updates it.
  Returns the list with removed duplicates.
  """
  no_dupes = []
  [no_dupes.append(e) for e in lst if e not in no_dupes]
  lst = no_dupes
  return lst 

def list_from_dicts(list_of_dicts, prim_key, sec_key=None):
  """
  From a list of dictionaries with homogenous keys, returns a 
  list of the values at a specified key and optional second key.
  Ex:
    list_of_dicts = [{"X": 1}, {"X": 2}, {"X": 3}]
    prim_key = "X"
    sec_key = None
      => [1,2,3]
  """
  new_list = []
  for e in list_of_dicts:
    new_list.append(e[prim_key][sec_key]) if sec_key \
      else new_list.append(e[prim_key])
  return new_list

def dict_to_tuple_list(dictionary):
  """
  Returns a list of tuples where each tuple
  represents a key, value pair from the input
  dictionary.
  """
  res = []
  for key, val in dictionary.items():
    res.append((key, val))
  return res

def max_key(dictionary):
  """
  Returns the key in the dictionary whose value
  is the maximum of all values in the dictionary.
  In the case of a tie, return one of the max keys
  at random.
  Ex:
    dictionary={"X": 4, "Y": 1, "Z": 4, "W": 2}
      => "X" or "Z" (randomly choice)
  """
  max_val = -math.inf
  keys = []
  for key, val in dictionary.items():
    if val is None:
      continue
    if val > max_val:
      max_val = val
      keys = [key]
    elif val == max_val:
      keys.append(key)
  return random.choice(keys) if keys else None

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

def num_matches(dict1, dict2):
  num = 0
  for key in dict1:
    if key in dict2 and dict1[key] == dict2[key]:
      num += 1
  return num

def dict_from_tuples(tuples):
  res = {}
  for tup in tuples:
    res[tup[0]] = tup[1]
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

def is_entry_in_dicts(dicts, entry):
  for d in dicts:
    for i in range(len(d)):
      key = list(d.keys())[i]
      if i == len(d) - 1:
        if d[key] == entry[key]:
          return True
      if d[key] != entry[key]:
        break
  return False

def avg(lst):
  """
  Returns the average/mean/expected value of a list
  of numbers
  """
  return sum(lst) / len(lst)

class Counter(dict):
  def __getitem__(self, idx):
    self.setdefault(idx, 0)
    return dict.__getitem__(self, idx)
  
  def copy(self):
    return Counter(dict.copy(self))
  
class CounterOne(dict):
  def __getitem__(self, idx):
    self.setdefault(idx, 1)
    return dict.__getitem__(self, idx)
  
  def copy(self):
    return CounterOne(dict.copy(self))

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

  
