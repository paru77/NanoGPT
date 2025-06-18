import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", 'r', encoding="utf-8") as f:
    text=f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)
print("".join(chars))
print(vocab_size) 

stoi={ch:i for i,ch in enumerate(chars)}
iots={i:ch for i, ch in enumerate(chars)}


encode= lambda s: [stoi[ch] for ch in s]
decode= lambda d:"".join([ iots[i] for i in d])


data=torch.tensor(encode(text), dtype=torch.long)
print(data.shape,data.dtype)
print(data[:100])

n=int(0.9*len(data))
train_data=data[:n]
test_data=data[n:]
print("Train data size =", len(train_data))
print("Test data size = ", len(test_data))

block_size=8
batch_size=32
max_iter=3000
eval_interval=300
learning_rate=1e-2
device="cuda" if torch.cuda.is_available() else "cpu"
eval_iter=200
n_embd=32


x=data[:block_size]
y=data[1:block_size+1]
print("Sequence",x)
for i in range(block_size):
    context=x[:i+1]
    target=y[i]
    print(f"when context is {context} the target is {target}")


# creatig the batch for parallel processing for each block size
torch.manual_seed(1337)      #just to replicate the same numbers 

def createBatch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    # print("Random generated number=",ix)
    x=torch.stack([data[i:block_size+i] for i in ix])
    y=torch.stack([data[i+1:block_size+i+1] for i in ix])
    x.to(device),y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out={}
    m.eval()
    for split in["train","val"]:
        losses=torch.zeros(eval_iter)
        for k in range(eval_iter):
            x,y=createBatch("train")
            logits,loss=m(x,y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    m.train()
    return out

# Bigram Langauge model, performs fine but dosen't give great results.
torch.manual_seed(1337) 

class BigramLangaugeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size, n_embd)
        self.lm_head=nn.linear(n_embd,vocab_size)
    
    def forward(self, idx,target=None):
        embedding=self.token_embedding_table(idx)    #(batchsize,blocksize,n_embd)
        logits= self.lm_head(embedding)                 #(batchsize,blocksize,Vocab_size)
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

            

m=BigramLangaugeModel()
m.to(device)

optimizer= torch.optim.AdamW(m.parameters(),lr=learning_rate)

for step in range(max_iter):

    if step%eval_interval==0:
        losses= estimate_loss()
        print(f"step {step}: train loss {losses["train"]} and val loss {losses["val"]}")
    
    xb,yb= createBatch("train")
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context=torch.zeros((1,1),dtype=torch.long, device= device)
print("Results of generation ", decode(m.generate(context, max_length=500)[0].tolist()))



