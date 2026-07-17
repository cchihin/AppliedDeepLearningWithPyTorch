import torch
import torch.nn as nn

batch = 1
tokens = 5
embed_size = 128

heads = 8

head_dim = embed_size // heads

assert(head_dim * heads == embed_size), "Head dims * number of heads is not equal to embed_size"


# (b, tokens, embed_size)
W = torch.rand(batch, tokens, embed_size)
print(f'Input shape: {W.shape}')

# Splitting heads: (b, tokens, heads, head_dim)
# And then transposing into (b, heads, tokens, head_dim) for @ to work
v_proj = nn.Linear(embed_size, embed_size)
k_proj = nn.Linear(embed_size, embed_size)
q_proj = nn.Linear(embed_size, embed_size)

V = v_proj(W).view(batch, tokens, heads, head_dim).transpose(1,2)
Q = q_proj(W).view(batch, tokens, heads, head_dim).transpose(1,2)
K = k_proj(W).view(batch, tokens, heads, head_dim).transpose(1,2)

# Applying QK^T: (b, h, t, hd) x (b, h, hd, t) = (b, h, t, t)
scores = Q @ K.transpose(2,3) / (head_dim ** 0.5)

# Apply masking to the scores
padding_mask = None
if padding_mask is not None:
    padding_mask = padding_mask.broadcast[:, 1, 1, :]
    scorces = scores.mask_fill(~padding_mask, -torch.inf)

casual = torch.tril(torch.ones(tokens, tokens))
decode = False
if decode:
    scores = scores.mask_fill(casual.broadcast[1, 1, :, :], -torch.inf)

weights = scores.softmax(dim=-1) # Applying softmax to the last dime


# Applying dropout
dropout = nn.Dropout(0.2)
z = dropout(weights) @ V

# out: (b, h, t, t) x (b, h, t, hd) = (b, h, t, hd) -> (b, t, h, hd) 
z = z.transpose(1,2)

# out_merge = (b, t, embed_size)
out_proj = nn.Linear(embed_size, embed_size)
out = out_proj(z.reshape(batch, tokens, embed_size))

print(f'Output shape: {out.shape}')
