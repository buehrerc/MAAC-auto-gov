from utils.policies import BasePolicy
import torch.nn.functional as F
from utils.misc import onehot_from_logits, epsilon_greedy


class CustomDiscretePolicy(BasePolicy):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        super(CustomDiscretePolicy, self).__init__(input_dim=input_dim, out_dim=output_dim, **kwargs)
        self.i = 0

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False, exploration_rate=0.1):
        out = super(CustomDiscretePolicy, self).forward(obs)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = epsilon_greedy(probs, epsilon=exploration_rate, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        self.i += 1
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out ** 2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets
