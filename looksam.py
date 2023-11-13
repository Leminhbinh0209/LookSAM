import torch
from collections import defaultdict

class LookSAM:
    def __init__(self, optimizer, model, rho=0.1, alpha=0.7):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.alpha = alpha
        self.perturb_eps = 1e-12
        self.state = defaultdict(dict)
        
    @torch.no_grad()
    
    def ascent_step(self):
        grad_norm = self._grad_norm()
        for n, p in self.model.named_parameters():
            scale = (self.rho  / (grad_norm + self.perturb_eps))
            if p.grad is None:
                continue
            eps = self.state[n].get("eps")
            old_grad = self.state[n].get("old_grad")
            vertical_grad = self.state[n].get("vertical_grad")
            
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[n]["eps"] = eps
                
            if old_grad is None:
                old_grad = torch.clone(p).detach()
                self.state[n]["old_grad"] = torch.clone(p).detach() # For the decomposing later
            
            if vertical_grad is None:
                vertical_grad = torch.clone(p).detach()
                self.state[n]["vertical_grad"] = torch.clone(p).detach() # For the decomposing later
                vertical_grad = p.grad[...]
            else:
                # already have from previuos iteration
                pass

            eps[...] = p.grad[...]
            old_grad[...] = p.grad[...]
            
        
            eps.mul_(scale)
            p.add_(eps)

        self.optimizer.zero_grad()
        
    @torch.no_grad()
    def _grad_norm(self, by=None):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        grads = []
        if not by:
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                grads.append(torch.norm(p.grad, p=2))
            grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        else:
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                grads.append(torch.norm(self.state[n][by], p=2))
            grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        return grad_norm   
    
    

    @torch.no_grad()
    def decompose_grad(self, project=False):
        """
        project: bool = True: t%k = 0 

        project = True: t%k != 0, don't need to calulate loss as perturbation step

        """
        if  project:
            # Add orginal gradient to the model
            inner_prod = 0.0
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                inner_prod += torch.sum(
                self.state[n]["old_grad"] * p.grad.data
                )

            # get norm
            new_grad_norm = self._grad_norm()
            old_grad_norm = self._grad_norm(by='old_grad')

            # get cosine
            cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

            # gradient decomposition
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                self.state[n]["vertical_grad"] = p.grad.data - new_grad_norm * cosine * self.state[n]["old_grad"] / (old_grad_norm + self.perturb_eps)
                # DO NOTHING with perturbed gradient
        else:
            cur_grad_norm = self._grad_norm()
            v_grad_norm  = self._grad_norm(by='vertical_grad')

            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                p.grad.data.add_(cur_grad_norm / (v_grad_norm + self.perturb_eps) *  self.state[n]["vertical_grad"], alpha=self.alpha)

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[n]["eps"])
