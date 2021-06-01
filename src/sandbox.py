import util
from cgm import CausalGraph
import os

if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + \
        'C:/Program Files/Graphviz/bin/'
    c1 = CausalGraph(nodes=util.get_nodes_from_edges([("X", "Y")]), edges=[
                     ("X", "Y")], latent_edges=[("X", "Y")])
    # c1.draw_model(True)
    print(c1.get_parents("X"))
    # c1.draw().render('output/no_do_X.gv', view=True)
    # c1.do("X").draw().render('output/do_X.gv', view=True)
    print(c1.do("X").get_parents("X"))
