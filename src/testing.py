from causalgraphicalmodels import CausalGraphicalModel

takeone = CausalGraphicalModel(
  nodes=["W","X","Y","Z"],
  edges=[("W","X"), ("W","Y"), ("X","Z"), ("Z","Y")]
)
takeone.draw().render('test-output/test-model.gv', view=False)