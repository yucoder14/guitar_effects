import torch 
import math

# not the most efficient way to do so, but in the way that my mind thinks about things
N = 5
model_dim = 4
batch_size = 10
key_dim = 2
num_heads = 2
num_batch = 10
value_dim = model_dim // num_heads
mask = torch.tensor([[0 if i>= j else -torch.inf for j in range(N)] for i in range(N)])

X = torch.ones((num_batch, N, model_dim))
X[:,1] = 2
X[:,:,2] = 3
print(X) # should not be symmetrical for demonstration purposes
single_head = torch.ones((model_dim, key_dim))
single_head_v = torch.ones((model_dim, value_dim))
W_Q = torch.cat([single_head * i for i in range(2, 2 + num_heads)], dim=1)
W_K = torch.cat([single_head * i for i in range(4, 4 + num_heads)], dim=1)
W_V = torch.cat([single_head_v * i for i in range(6, 6 + num_heads)], dim=1)
W_O = torch.rand((model_dim, model_dim))
print(X.shape)
print(W_Q.shape)
Q = X@W_Q
K = X@W_K
V = X@W_V
print(Q.shape) # batch_size, N, key_dim * num_heads
print(Q.unbind()[0]) # essentially Q's from different heads concatenated to be next to each other "vertically"
Q_reshaped = Q.view(num_batch, N, num_heads, key_dim).transpose(1,2) 
K_reshaped = K.view(num_batch, N, num_heads, key_dim).transpose(1,2)
V_reshaped = V.view(num_batch, N, num_heads, value_dim).transpose(1,2)
print(Q_reshaped.unbind()[0])
print(Q.data_ptr() == Q_reshaped.data_ptr()) # this should be False, meaning reshape has created new tensor, which is not memory efficient
attention = Q_reshaped@K_reshaped.mT / math.sqrt(key_dim) + mask # mask broadcasting
print(attention.shape, V_reshaped.shape)
probs = torch.nn.functional.softmax(attention, dim=-1) 
values = probs@V_reshaped / 24 # just dividing by arbitrary number for ease of seeing value matrix of each head as a unified number
print(values.shape) # batch_size, num_heads, N, value_dim
# batch_size, num_heads, N, value_dim --> batch_size, num_heads, value_dim, N --> batch_size, num_heads*value_dim, N --> batch_size, N, num_heads*value_dim (model_dim)
values_cat = values.transpose(-2, -1).flatten(start_dim=1, end_dim=2).transpose(-2, -1)
# batch_size, num_heads, value_dim, N --> batch_size, N, num_heads, value_dim --> batch_size, N, num_heads*value_dim
values_cat_2 = values.transpose(1, 2).contiguous().view(batch_size, N, -1)
values_cat == values_cat_2
#values_cat@W_O