import torch as t
import torch.nn as nn
from attack_model import MEMBERINF

### DP-SGD
# Reference: https://github.com/heyyjudes/dp-sgd
class DP_SGD:
    """
        DP-SGD
    """

    def __init__(self, model, optim, device, loss=None, dp_sigma=1.0, 
                                l2_norm_clip=1.0, batch_size=64, indiv=False):
        """
            indiv: individual gradient update, loss should set reduce=False
        """
        self.noise_multiplier = dp_sigma/batch_size
        # self.dp_sigma = dp_sigma
        self.l2_norm_clip = l2_norm_clip
        self.batch_size = batch_size
        self.model = model
        self.optim = optim
        self.device = device
        self.indiv = indiv
        self.loss = loss

    def _clip_grads(self, grads): 
        # clip gradients 
        squared_norms = t.stack([(p.grad.view(1, -1) ** 2).sum(dim=1) 
                        for p in self.model.parameters() if p.grad is not None])
        grad_norm = t.sqrt(squared_norms.sum(dim=0))

        factor = self.l2_norm_clip/grad_norm
        factor = t.clamp(factor, max=1.0) 
        
        # # add to gradient vector 
        for g, p in zip(grads, self.model.parameters()):
            if p.grad is not None:
                if self.indiv:
                    g += (factor/self.batch_size)*p.grad.clone()
                else:
                    g += factor*p.grad.clone()

    def _save_grads(self, grads): 
        # save gradient vector in actual 
        for g, p in zip(grads, self.model.parameters()):
            if p.grad is not None:
                p.grad = t.add(
                    g, 
                    alpha=self.noise_multiplier * self.l2_norm_clip,
                    other=t.randn_like(g)
                    )

    def set_loss(self, loss):
        self.loss = loss
        
    def dp_sgd(self):
        grads = [t.zeros((*p.shape)).to(self.device) for p in self.model.parameters()] 

        if self.indiv:
            for i in range(self.batch_size):
                self.optim.zero_grad()
                self.loss[i].backward(retain_graph=True)
                self._clip_grads(grads)
        else:
            self.optim.zero_grad()
            self.loss.backward()
            self._clip_grads(grads)
        self._save_grads(grads)


class MIN_MAX:
    """
        Min-max
    """
    def __init__(self, model_cls, trainiter_inf, refiter_inf, device, lr=0.001, momentum=0.9, weight_decay=5e-4, num_step=5):
        self.model_inf = MEMBERINF().to(device)
        self.criterion_inf = nn.CrossEntropyLoss()

        self.optim_inf = t.optim.SGD(self.model_inf.parameters(), 
                                    lr=lr, momentum=momentum, 
                                    weight_decay=weight_decay)
        self.num_step = num_step
        self.model_cls = model_cls
        self.trainiter_inf = trainiter_inf
        self.refiter_inf = refiter_inf
        self.device = device

    def _inf_update(self):
        running_loss = 0.0

        for i in range(self.num_step):
            self.optim_inf.zero_grad()
            
            train_data, _ = self.trainiter_inf.__next__()
            train_data = train_data.to(self.device)
            out_train = self.model_cls.classify(train_data)
            out_inf_train = self.model_inf(out_train)
            train_y = t.LongTensor([1 for _ in range(len(train_data))]).to(self.device)

            ref_data, _ = self.refiter_inf.__next__()
            ref_data = ref_data.to(self.device)
            out_ref = self.model_cls.classify(ref_data)
            out_inf_ref = self.model_inf(out_ref)
            ref_y = t.LongTensor([0 for _ in range(len(train_data))]).to(self.device)

            loss = (self.criterion_inf(out_inf_train, train_y) 
                   + self.criterion_inf(out_inf_ref, ref_y))
            loss.backward()
            self.optim_inf.step()

            running_loss += loss.item()
        
        return running_loss

    def update_loss(self, logits, l_p_y_given_x):
        inf_loss = self._inf_update()
        print("mem inf loss:", inf_loss, end='\r')

        out_inf_train = self.model_inf(logits)
        train_y = t.LongTensor([1 for _ in range(len(logits))]).to(self.device)
        return self.criterion_inf(out_inf_train, train_y)
        