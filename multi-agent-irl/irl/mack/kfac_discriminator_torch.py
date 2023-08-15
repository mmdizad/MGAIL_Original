import torch
from torch import nn

disc_types = ['decentralized', 'centralized', 'single']

class Discriminator(nn.Module):
    def __init__(self, ob_spaces, ac_spaces,
                 nstack, index, disc_type='decentralized', hidden_size=128,
                 lr_rate=0.01, total_steps=50000):
        super(Discriminator, self).__init__()
        self.ob_spaces = ob_spaces
        self.ac_spaces = ac_spaces
        self.nstack = nstack
        self.index = index
        self.disc_type = disc_type
        self.hidden_size = hidden_size
        # TODO
        self.lr_rate = lr_rate
        # TODO
        self.total_steps = total_steps
        self.num_outputs = len(ob_spaces) if disc_type == 'centralized' else 1
        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        self.ob_shape = ob_space.shape[0] * nstack
        try:
            nact = ac_space.n
        except:
            nact = ac_space.shape[0]
        self.ac_shape = nact * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        try:
            self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
        except:
            self.all_ac_shape = sum([ac.shape[0] for ac in ac_spaces]) * nstack
        if disc_type == 'decentralized':
            self.disc = self.build_graph(input_shape=self.ob_shape + self.ac_shape, output_shape=self.num_outputs)
        elif disc_type == 'centralized':
            self.disc = self.build_graph(input_shape=self.all_ob_shape + self.all_ac_shape, output_shape=self.num_outputs)
        elif disc_type == 'single':
            self.disc = self.build_graph(input_shape=self.all_ob_shape + self.all_ac_shape,
                                         output_shape=self.num_outputs)
        else:
            assert False

    def build_graph(self, input_shape, output_shape):
        disc = nn.Sequential(
        nn.Linear(input_shape, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, output_shape)
        )
        return disc



