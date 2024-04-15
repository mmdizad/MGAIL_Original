import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl.acktr.utils_torch import fc

disc_types = ['decentralized', 'centralized', 'single']

class Discriminator(nn.Module):
    def __init__(self, ob_spaces, ac_spaces,
                 nstack, index, device, disc_type='decentralized', hidden_size=128,
                 learning_rate=0.01, max_grad_norm=0.5, weight_decay=1e-5):
        super(Discriminator, self).__init__()
        self.max_grad_norm = max_grad_norm
        self.hidden_size = hidden_size        
        self.num_outputs = len(ob_spaces) if disc_type == 'centralized' else 1
        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        self.ob_shape = ob_space.shape[0] * nstack
        self.ac_shape = ac_space.n * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
        self.device = device
        self.logsigmoid = torch.nn.LogSigmoid()
                
        if disc_type == 'decentralized':
            self.disc = self.build_graph(input_shape=self.ob_shape + self.ac_shape, output_shape=self.num_outputs)
        elif disc_type == 'centralized':
            self.disc = self.build_graph(input_shape=self.all_ob_shape + self.all_ac_shape, output_shape=self.num_outputs)
        elif disc_type == 'single':
            self.disc = self.build_graph(input_shape=self.all_ob_shape + self.all_ac_shape,
                                         output_shape=self.num_outputs)
        else:
            assert False
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5) # TODO scheduler support
        
    def build_graph(self, input_shape, output_shape):
        disc = nn.Sequential(
            fc(input_shape, self.hidden_size),
            nn.ReLU(),
            fc(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            fc(self.hidden_size, output_shape),
        )
        return disc
    
    
    def forward(self, ob, act):
        ob, act = torch.tensor(ob, dtype=torch.float32, requires_grad=True).to(self.device),\
            torch.tensor(act, dtype=torch.float32, requires_grad=True).to(self.device)
        score = self.disc(torch.cat([ob, act], dim=1))
        return score
    
    def train(self, ob_pi, act_pi, ob_exp, act_exp):
        loss_pi, loss_exp, back_loss = self.calculate_loss(ob_pi, act_pi, ob_exp, act_exp)
        loss = loss_pi + loss_exp
        self.optimizer.zero_grad()
        back_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss_pi.detach().cpu(), loss_exp.detach().cpu()
    
    def calculate_loss(self, ob_pi, act_pi, ob_exp, act_exp):
        logits_pi = self.forward(ob_pi, act_pi)
        logits_exp = self.forward(ob_exp, act_exp)
        logits = torch.cat([logits_pi, logits_exp], dim=0)
        
        labels_pi = torch.zeros((ob_pi.shape[0], 1), device=self.device)
        labels_exp = torch.ones((ob_exp.shape[0], 1), device=self.device)
        labels = torch.cat([labels_pi, labels_exp], dim=0)
        loss_pi = F.binary_cross_entropy_with_logits(logits_pi, labels_pi)
        loss_exp = F.binary_cross_entropy_with_logits(logits_exp, labels_exp)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return loss_pi, loss_exp, loss
    
    def get_reward(self, ob, act):
        score = self.forward(ob, act)
        reward = self.logsigmoid(score)
        return reward.detach().cpu().numpy()
    
    def save(self, save_path='disc_model_weights.pth'):
        torch.save(self.state_dict(), save_path) # TODO save optimizer parameters
        
        
    def load(self, load_path='disc_model_weights.pth'):
        self.load_state_dict(torch.load(load_path)) # TODO load optimizer parameters
        
        
