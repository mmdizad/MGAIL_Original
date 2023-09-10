import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl.acktr.utils_torch import fc

disc_types = ['decentralized', 'centralized', 'single']

class Discriminator(nn.Module):
    def __init__(self, ob_spaces, ac_spaces,
                 nstack, index, device, disc_type='decentralized', hidden_size=128,
                 learning_rate=0.01, total_steps=50000):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size        
        self.num_outputs = len(ob_spaces) if disc_type == 'centralized' else 1
        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        self.ob_shape = ob_space.shape[0] * nstack
        self.ac_shape = ac_space.n * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
        self.device = device
                
        if disc_type == 'decentralized':
            self.disc = self.build_graph(input_shape=self.ob_shape + self.ac_shape, output_shape=self.num_outputs)
        elif disc_type == 'centralized':
            self.disc = self.build_graph(input_shape=self.all_ob_shape + self.all_ac_shape, output_shape=self.num_outputs)
        elif disc_type == 'single':
            self.disc = self.build_graph(input_shape=self.all_ob_shape + self.all_ac_shape,
                                         output_shape=self.num_outputs)
        else:
            assert False
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate) # TODO scheduler support
        
    def build_graph(self, input_shape, output_shape):
        disc = nn.Sequential(
            fc(input_shape, self.hidden_size),
            nn.ReLU(),
            fc(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            fc(self.hidden_size, output_shape)
        )
        return disc
    
    
    def forward(self, ob, act):
        ob, act = torch.tensor(ob, dtype=torch.float32).to(self.device),\
            torch.tensor(act, dtype=torch.float32).to(self.device)
        score = self.disc(torch.cat([ob, act], dim=1))
        return score
    
    def train(self, ob_pi, act_pi, ob_exp, act_exp):
        loss_pi, loss_exp = self.calculate_loss(ob_pi, act_pi, ob_exp, act_exp)
        loss = loss_pi + loss_exp
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_pi.detach().cpu(), loss_exp.detach().cpu()
    
    def calculate_loss(self, ob_pi, act_pi, ob_exp, act_exp):
        logits_pi = self.forward(ob_pi, act_pi)
        logits_exp = self.forward(ob_exp, act_exp)
        labels_pi = torch.zeros((ob_pi.shape[0], 1))
        labels_exp = torch.ones((ob_exp.shape[0], 1))
        loss_pi = F.binary_cross_entropy_with_logits(logits_pi, labels_pi)
        loss_exp = F.binary_cross_entropy_with_logits(logits_exp, labels_exp)

        #  maximize E_{\pi} log(D) + E_{exp} log(-D)
        # loss_pi = torch.mean(-F.logsigmoid(logits_pi))
        # loss_exp = torch.mean(-F.logsigmoid(-logits_exp))
        
        return loss_pi, loss_exp
    
    def get_reward(self, ob, act):
        score = self.forward(ob, act)
        reward = torch.log(torch.sigmoid(score) + 1e-10)
        return reward.detach().cpu().numpy()
    
    def save(self, save_path='disc_model_weights.pth'):
        torch.save(self.state_dict(), save_path) # TODO save optimizer parameters
        
        
    def load(self, load_path='disc_model_weights.pth'):
        self.load_state_dict(torch.load(load_path)) # TODO load optimizer parameters
        
        
