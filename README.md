
<h1 align="center"><b>LookSAM Optimizer</b></h1>
<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/Leminhbinh0209/LookSAM?style=for-the-badge" height="25"  onmouseover="this.height='60'" onmouseout="this.height='25'" >
</br>
<h3 align="center"><b>Towards Efficient and Scalable Sharpness-Aware Minimization </b></h3>
<a href="https://arxiv.org/pdf/2203.02714.pdf" a> [Paper, CVPR 2022]</a>   

## How to use 
```python
ce_loss = nn.CrossEntropyLoss()
base_optimizer = torch.optim.SGD(model.parameters())
step_k = 5
minimizer = LookSAM(base_optimizer, model=model, rho=0.1, alpha=0.7)

for batch_idx, (input, label) in enumerate(dataloader):

    logit = model(input)
    loss = ce_loss(logit, label)
    loss.backward()
    
    # Perform algorithm 1
    # Calculate loss at ascending step
    if batch_idx%step_k==0:
        minimizer.ascent_step()
        as_logit = model(input)
        as_loss = ce_loss(as_logit, label)
        as_loss.backward()

    
    minimizer.decompose_grad(batch_idx%step_k==0)

    # Perform descending step
    if batch_idx%step_k==0:
        minimizer.descent_step()

    # Update the weights
    optimizer.step() 
    optimizer.zero_grad()

```

Credits to [GSAM implementation](https://github.com/juntang-zhuang/GSAM)

#
*Star (⭐) if you find it useful*  
