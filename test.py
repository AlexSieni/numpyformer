# train a model based on these
from nn.modules.embedding import Embedding
from tensor import Tensor, as_tensor
import numpy as np
from nn import Linear, LayerNorm, Tanh, AttentionHead


words = open("input.txt", 'r').read()
vocab = sorted(list(set(''.join(words))) + ['.'])
vocab_size = len(vocab)
batch_size = 32
block_size = 4
emb_size = 8

stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for i,s in enumerate(vocab)}
encode = lambda w: [stoi[c] for c in w]
decode = lambda w: ''.join([itos[c] for c in w])

data = encode(words)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = np.array(data[:n]) 
val_data = np.array(data[n:])    

# we will train an n-gram model
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - block_size, size=(batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    x = Tensor(x, requires_grad=False, dtype=np.int32)
    y = Tensor(y, requires_grad=False, dtype=np.int32)
    return x, y


# n = int(.9 * len(words))
# Xtr, Ytr = get_batch('train')
# Xval, Yval = get_batch('val')

hidden = 16

layers = [
    AttentionHead(emb_size=emb_size), LayerNorm(emb_size, symbol="Layer"),
    Linear(emb_size, hidden, symbol="Linear 1"), LayerNorm(hidden, symbol="Layer"), Tanh(symbol="Tanh"),
    Linear(hidden, vocab_size)
]

# print(f"Xtr:{Xtr[:5]}, Ytr:{Ytr[:5]}")

# Xtrc = Xtr.copy()

embeddings = Embedding(vocab_size, emb_size)
position_emb = Embedding(block_size, emb_size)

grads = dict()
for layer in layers:
    for p in layer.params():
        grads[p] = []

for p in embeddings.params():
    grads[p] = []

for p in position_emb.params():
    grads[p] = []

lr = 0.001  # Learning rate

for iteration in range(1000):
    x, y = get_batch('train')
    x0 = x
    x = embeddings(x0)
    pos_x = position_emb(np.arange(x0.shape[1]))
    x = x + pos_x  # Broadcasting: (B, T, emb_size) + (T, emb_size)
    for _, layer in enumerate(layers):
        x = layer.forward(x)
    B, T, C = x.shape
    logits = x.view(B*T, C)
    loss = logits.cross_entropy(y.view(B*T))
    # print(f"Iteration {iteration}, Loss: {loss.data}")
    loss.backward()

    # Include embedding parameters in gradient updates
    params = [p for layer in layers for p in layer.params()]
    params += embeddings.params() + position_emb.params()
    
    for p in params:
        p.data -= lr * p.grad

    for p in params:
        if p.grad is not None:
            p.grad.fill(0.0)

print(loss.data)
print(len(vocab))

import matplotlib.pyplot as plt

# Extract final gradients for each parameter
fin_mean = []
fin_std = []

# Layer parameters
for i, layer in enumerate(layers):
    fin_mean.append(layer.out.data.mean())
    fin_std.append(layer.out.data.std())



# Create a single plot with distributions
plt.figure(figsize=(14, 8))
x_range = np.linspace(-2, 2, 200)  # Adjust range as needed

for i, (mean, std) in enumerate(zip(fin_mean, fin_std)):
    y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean)/std)**2)
    plt.plot(x_range, y, label=f'Layer {i}')

plt.title('Output Distribution per Layer (Normal Approximation)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
