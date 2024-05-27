import numpy as np
import torch
import torch.nn as nn
from torch import distributions as dist

from .encoder import SimplePointnet, ResnetPointnet


class AbsLayer(nn.Module):
    def __init__(self):
        super(AbsLayer, self).__init__()

    def forward(self, x):
        return torch.abs(x)


def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Decoder(nn.Module):
    def __init__(self, udf=False):
        super(Decoder, self).__init__()
        self.nl = nn.Identity() if not udf else AbsLayer()

    def forward(self, *args, **kwargs):
        res = self.fc_block(*args, **kwargs)
        res = self.nl(res)
        return res


class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=-1)

        return tuple(hiddens)


class Network(nn.Module):
    def __init__(self, latent_size=0, in_dim=3, decoder_hidden_dim=256, nl='sine', decoder_n_hidden_layers=4,
                 init_type='siren', sphere_init_params=[1.6, 1.0], udf=False, vae=False):
        super().__init__()
        self.latent_size = latent_size
        self.vae = vae

        self.encoder = SimplePointnet(c_dim=latent_size, dim=3) if latent_size > 0 and vae == True else None
        # self.encoder = ResnetPointnet(c_dim=latent_size, dim=3) if latent_size > 0 and vae == True else None
        self.modulator = Modulator(
            dim_in=latent_size,
            dim_hidden=decoder_hidden_dim,
            num_layers=decoder_n_hidden_layers + 1  # +1 for input layer
        ) if latent_size > 0 else None

        self.init_type = init_type
        self.decoder = Decoder(udf=udf)
        self.decoder.fc_block = FCBlock(in_dim, 1, num_hidden_layers=decoder_n_hidden_layers,
                                        hidden_features=decoder_hidden_dim,
                                        outermost_linear=True, nonlinearity=nl, init_type=init_type,
                                        sphere_init_params=sphere_init_params)  # SIREN decoder

    def forward(self, non_mnfld_pnts, mnfld_pnts=None, near_points=None, only_nonmnfld=False, latent=None):
        batch_size = non_mnfld_pnts.shape[0]
        if self.latent_size > 0 and self.encoder is not None:
            # encoder
            q_latent_mean, q_latent_std = self.encoder(mnfld_pnts)
            q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent = q_z.rsample()
            latent_reg = 1.0e-3 * (q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1))
            # modulate
            modulate = exists(self.modulator)
            mods = self.modulator(latent) if modulate else None
        elif self.latent_size > 0:
            modulate = exists(self.modulator)
            mods = self.modulator(latent) if modulate else None
            latent_reg = 1e-3 * latent.norm(-1).mean()
        else:
            latent = None
            latent_reg = None
            mods = None
        if mnfld_pnts is not None and not only_nonmnfld:
            manifold_pnts_pred = self.decoder(mnfld_pnts, mods)
        else:
            manifold_pnts_pred = None
        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts, mods)

        near_points_pred = None
        if near_points is not None:
            near_points_pred = self.decoder(near_points, mods)

        return {"manifold_pnts_pred": manifold_pnts_pred,
                "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
                'near_points_pred': near_points_pred,
                "latent_reg": latent_reg,
                }

    def get_latent_mods(self, mnfld_pnts=None, latent=None, rand_predict=True):
        mods = None
        if self.vae:
            if rand_predict:
                q_latent_mean, q_latent_std = self.encoder(mnfld_pnts)
                q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
                latent = q_z.rsample()
            else:
                latent, _ = self.encoder(mnfld_pnts)
        modulate = exists(self.modulator)
        mods = self.modulator(latent) if modulate else None
        return mods


class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren',
                 sphere_init_params=[1.6, 1.0]):
        super().__init__()
        print("decoder initialising with {} and {}".format(nonlinearity, init_type))

        self.first_layer_init = None
        self.sphere_init_params = sphere_init_params
        self.init_type = init_type

        nl_dict = {'sine': Sine(), 'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(beta=100),
                   'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
        nl = nl_dict[nonlinearity]

        self.net = []
        self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(nn.Linear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features), nl))

        self.net = nn.Sequential(*self.net)

        if init_type == 'siren':
            self.net.apply(sine_init)
            self.net[0].apply(first_layer_sine_init)

        elif init_type == 'geometric_sine':
            self.net.apply(geom_sine_init)
            self.net[0].apply(first_layer_geom_sine_init)
            self.net[-2].apply(second_last_layer_geom_sine_init)
            self.net[-1].apply(last_layer_geom_sine_init)

        elif init_type == 'mfgi':
            self.net.apply(geom_sine_init)
            self.net[0].apply(first_layer_mfgi_init)
            self.net[1].apply(second_layer_mfgi_init)
            self.net[-2].apply(second_last_layer_geom_sine_init)
            self.net[-1].apply(last_layer_geom_sine_init)

        elif init_type == 'geometric_relu':
            self.net.apply(geom_relu_init)
            self.net[-1].apply(geom_relu_last_layers_init)

    def forward(self, coords, mods=None):
        mods = cast_tuple(mods, len(self.net))
        x = coords

        for layer, mod in zip(self.net, mods):
            x = layer(x)
            if exists(mod):
                if mod.shape[1] != 1:
                    mod = mod[:, None, :]
                x = x * mod
        if mods[0] is not None:
            x = self.net[-1](x)  # last layer

        if self.init_type == 'mfgi' or self.init_type == 'geometric_sine':
            radius, scaling = self.sphere_init_params
            output = torch.sign(x) * torch.sqrt(x.abs() + 1e-8)
            output -= radius  # 1.6
            output *= scaling  # 1.0

        return x


class Sine(nn.Module):
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


################################# SIREN's initialization ###################################
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


################################# sine geometric initialization ###################################

def geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30


def first_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30


def second_last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            assert m.weight.shape == (num_output, num_output)
            m.weight.data = 0.5 * np.pi * torch.eye(num_output) + 0.001 * torch.randn(num_output, num_output)
            m.bias.data = 0.5 * np.pi * torch.ones(num_output, ) + 0.001 * torch.randn(num_output)
            m.weight.data /= 30
            m.bias.data /= 30


def last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (1, num_input)
            assert m.bias.shape == (1,)
            # m.weight.data = -1 * torch.ones(1, num_input) + 0.001 * torch.randn(num_input)
            m.weight.data = -1 * torch.ones(1, num_input) + 0.00001 * torch.randn(num_input)
            m.bias.data = torch.zeros(1) + num_input


################################# multi frequency geometric initialization ###################################
periods = [1, 30]  # Number of periods of sine the values of each section of the output vector should hit
# periods = [1, 60] # Number of periods of sine the values of each section of the output vector should hit
portion_per_period = np.array([0.25, 0.75])  # Portion of values per section/period


def first_layer_mfgi_init(m):
    global periods
    global portion_per_period
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            num_output = m.weight.size(0)
            num_per_period = (portion_per_period * num_output).astype(int)  # Number of values per section/period
            assert len(periods) == len(num_per_period)
            assert sum(num_per_period) == num_output
            weights = []
            for i in range(0, len(periods)):
                period = periods[i]
                num = num_per_period[i]
                scale = 30 / period
                weights.append(torch.zeros(num, num_input).uniform_(-np.sqrt(3 / num_input) / scale,
                                                                    np.sqrt(3 / num_input) / scale))
            W0_new = torch.cat(weights, axis=0)
            m.weight.data = W0_new


def second_layer_mfgi_init(m):
    global portion_per_period
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (num_input, num_input)
            num_per_period = (portion_per_period * num_input).astype(int)  # Number of values per section/period
            k = num_per_period[0]  # the portion that only hits the first period
            # W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.00001
            W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input),
                                                                np.sqrt(3 / num_input) / 30) * 0.0005
            W1_new_1 = torch.zeros(k, k).uniform_(-np.sqrt(3 / num_input) / 30, np.sqrt(3 / num_input) / 30)
            W1_new[:k, :k] = W1_new_1
            m.weight.data = W1_new


################################# geometric initialization used in SAL and IGR ###################################
def geom_relu_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            out_dims = m.out_features

            m.weight.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dims))
            m.bias.data = torch.zeros_like(m.bias.data)


def geom_relu_last_layers_init(m):
    radius_init = 1
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.normal_(mean=np.sqrt(np.pi) / np.sqrt(num_input), std=0.00001)
            m.bias.data = torch.Tensor([-radius_init])
