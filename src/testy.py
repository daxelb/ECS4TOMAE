class Episode:
  def __init__(self):
    return
    
  def run(self):
    for self.i in range(10):
      print(self.i + self.j)
      
  def run2(self):
    for self.j in range(0, 100, 10):
      self.run()
      
  def list_from_dicts(self, list_of_dicts, prim_key, sec_key=None):
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
      
if __name__ == "__main__":
  # episode = Episode()
  # print(episode.list_from_dicts([{"adjust": {"0": 1, "2": 2}, "deaf": {"0": 1, "2": 0}}, {"adjust": {"0": 5, "2": 4}, "deaf": {"0": 3, "2": 2}}], "adjust", "0"))
  import pandas as pd
  d = {"a": [1,2,3], "b": [0,2,4], "c": [10,20,30]}
  df = pd.DataFrame(data=d.values(),columns=["A","B","C"]).sum()
  print(df)
  