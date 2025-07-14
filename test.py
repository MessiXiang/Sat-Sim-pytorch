import torch
from satsim.utils.matrix_support import to_rotation_matrix, to_rotation_matrix

a = torch.rand(3)
a_1 = to_rotation_matrix(a)
a_2 = to_rotation_matrix(a)
print(a_1)
print(a_2)
