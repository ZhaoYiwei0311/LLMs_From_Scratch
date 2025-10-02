import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_attention import MultiHeadAttention,MultiHeadAttentionWrapper

inputs = torch.tensor( 
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(789)
batch = torch.stack((inputs, inputs), dim=0)

torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs) 
print("context_vecs.shape:", context_vecs.shape)

# a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573], 
#                     [0.8993, 0.0390, 0.9268, 0.7388], 
#                     [0.7179, 0.7058, 0.9156, 0.4340]], 
#                    [[0.0772, 0.3565, 0.1479, 0.5331], 
#                     [0.4066, 0.2318, 0.4545, 0.9737], 
#                     [0.4606, 0.5159, 0.4220, 0.5786]]]])
# print(a @ a.transpose(2, 3))

torch.manual_seed(123) 
batch_size, context_length, d_in = batch.shape 
d_out = 2 
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2) 
context_vecs = mha(batch) 
print(context_vecs) 
print("context_vecs.shape:", context_vecs.shape)