

# Calculate the average node degree in the training data

def avg_d(dataset):
  
  D = []
  
  for i in range(len(dataset)):

      adj = to_dense_adj(dataset[i].edge_index)[0]

      deg = adj.sum(axis=1, keepdim=True) # Degree of nodes, shape [N, 1]

      D.append(deg.squeeze())

  D = torch.cat(D, dim = 0)

  avg_d  = dict(lin=torch.mean(D),
          exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
          log=torch.mean(torch.log(D + 1)))


  return D, avg_d
