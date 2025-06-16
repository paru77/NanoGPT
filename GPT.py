import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", 'r', encoding="utf-8") as f:
    text=f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)
print("".join(chars))
print(vocab_size)   #code box size(very small)

stoi={ch:i for i,ch in enumerate(chars)}
iots={i:ch for i, ch in enumerate(chars)}
# print("Stoi",stoi)
# print("iots",iots)

encode= lambda s: [stoi[ch] for ch in s]
decode= lambda d:"".join([ iots[i] for i in d])

# print(encode("Hello world"))
# print(decode(encode("Hello world")))

data=torch.tensor(encode(text), dtype=torch.long)
print(data.shape,data.dtype)
print(data[:100])

n=int(0.9*len(data))
train_data=data[:n]
test_data=data[n:]
print("Train data size =", len(train_data))
print("Test data size = ", len(test_data))


# to show the the input context should be and how the target should be like
block_size=8
x=data[:block_size]
y=data[1:block_size+1]
print("Sequence",x)
for i in range(block_size):
    context=x[:i+1]
    target=y[i]
    print(f"when context is {context} the target is {target}")

batch_size=4
# creatig the batch for parallel processing for each block size
torch.manual_seed(1337)      #just to replicate the same numbers 

def createBatch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    # print("Random generated number=",ix)
    x=torch.stack([data[i:block_size+i] for i in ix])
    y=torch.stack([data[i+1:block_size+i+1] for i in ix])
    return x,y

xb,yb=createBatch("train")
print("Training batch",xb)
print("Traget of the training batch",yb)


print("----------------Printing the context and target------------------------")
for i in range(batch_size):
    for j in range(block_size):
        x=xb[i][:j+1]
        y=yb[i][j]
        print(f"When the context is {x} the target is {y}")

torch.manual_seed(1337) 
class BigramLangaugeModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size, vocab_size)
        print(self.token_embedding_table)
    
    def forward(self, idx,target=None):
        logits=self.token_embedding_table(idx)    #(batchsize,blocksize,vocab length)
        if target==None:
            loss=None
        else:
            B,T,V=logits.shape
            logits=logits.view(B*T,V)
            target=target.view(B*T)
            loss=F.cross_entropy(logits,target)
        
        return logits ,loss

    def generate(self, idx, max_length):
        for _ in range(max_length):
            logits,loss=self(idx)
            logits=logits[:,-1,:]
            probs=F.softmax(logits, dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx

            

m=BigramLangaugeModel(vocab_size)
logits,loss=m(xb,yb)
print(loss)

print(decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long), max_length=50)[0].tolist()))

optimizer= torch.optim.AdamW(m.parameters(),lr=1e-3)
batch_size=32
for step in range(100):
    xb,yb= createBatch("train")
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())