# LookSAM
Unofficial Implementation of "Towards Efficient and Scalable Sharpness-Aware Minimization"

# How to use 
```
ce_loss = nn.CrossEntropyLoss()
base_optimizer = torch.optim.SGD(model.parameters())
step_k = 5
minimizer = LookSAM(base_optimizer, model=model, rho=0.1, alpha=0.7)

for batch_idx, (input, label) in enumerate(dataloader):
    # Perform ascending step
    logit = model(input)
    loss = ce_loss(logit, label)
    loss.backward()
    minimizer.ascent_step()

    # Perform algorithm 1
    # Calculate loss at ascending step
    if batch_idx%step_k==0
        as_logit = model(input)
        as_loss = ce_loss(as_logit, label)
        as_loss.backward()

    
    minimizer.decompose_grad(batch_idx%step_k==0)

    # Perform descending step
    minimizer.descent_step()

    # Update the weights
    optimizer.step() 
    optimizer.zero_grad()

```

Credits to [GSAM implementation](https://github.com/juntang-zhuang/GSAM)

#
*Star (⭐) if you find it useful*  
