import torch
from graph_transformer_pytorch import GraphTransformer

model = GraphTransformer(
    dim = 256,
    depth = 6,
          # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
    with_feedforwards = True,   # whether to add a feedforward after each attention layer, suggested by literature to be needed
    gated_residual = True,      # to use the gated residual to prevent over-smoothing
    rel_pos_emb = True          # set to True if the nodes are ordered, default to False
)

nodes = torch.randn(1,32, 256)
edges = torch.randn(4, 256)
mask = torch.ones(1, 32).bool()

nodes, edges = model(nodes, edges, mask = mask)

print(nodes.shape)
print(edges.shape)