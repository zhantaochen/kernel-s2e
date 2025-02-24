from collections import namedtuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import lightning as L

from .siren import SirenNet

def get_ff_params():
    ff = namedtuple(
        'ff', 
        ['A0', 'a0', 'B0', 'b0', 'C0', 'c0', 'D0',
         'A4', 'a4', 'B4', 'b4', 'C4', 'c4', 'D4']
        )(0.0163, 35.8826, 0.3916, 13.2233, 0.6052, 4.3388, -0.0133,
          -0.3803, 10.4033, 0.2838, 3.3780, 0.2108, 1.1036, 0.0050)
    return ff

def scale_tensor(tensor, bounds_init, bounds_fnal=(-1., 1.)):
    min_init, max_init = bounds_init
    min_fnal, max_fnal = bounds_fnal
    return ((tensor - min_init) * (max_fnal - min_fnal) / (max_init - min_init)) + min_fnal

class SpecNeuralRepr(L.LightningModule):
    def __init__(
        self, 
        scale_dict={
            'J' : [(20, 40), (0, 0.5)], 
            'Jp': [(-5,  5), (0, 0.5)], 
            'w' : [(0, 150), (0, 0.5)]}
    ):
        super().__init__()
        self.save_hyperparameters()
        # lattice constants
        self.latt_const = namedtuple('latt_const', ['a', 'c'])(3.89, 12.55)
        # form factor parameters
        self.ff = get_ff_params()
        # networks
        self.Syy_net = torch.nn.Sequential(
            SirenNet(
                dim_in = 5,
                dim_hidden = 256,
                dim_out = 256,
                num_layers = 3,
                w0_initial = 30.,
                final_activation = torch.nn.ReLU()
            ),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.Szz_net = torch.nn.Sequential(
            SirenNet(
                dim_in = 5,
                dim_hidden = 256,
                dim_out = 256,
                num_layers = 3,
                w0_initial = 30.,
                final_activation = torch.nn.ReLU()
            ),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.scale_dict = scale_dict
 
    def prepare_input(self, x):
        shape = x.shape[:-1]
        x = x.view(-1, x.size(-1)).to(self.dtype)
        for key in self.scale_dict.keys():
            if key == 'w':
                i = 2
            elif key == 'J':
                i = 3
            elif key == 'Jp':
                i = 4
            x[:,i] = scale_tensor(x[:,i], *self.scale_dict[key])
        return x.view(shape+(x.shape[-1],))
        
    def forward(self, x_raw, l=None, Syy=None, Szz=None):
        """
        x_raw: (..., 5)
        the 1st and 2nd are the reciprocal lattice vectors (h,k)
        the 3rd dimension is the energy transfer w
        the 4th and 5th dimensions are the query parameters
        """
        # avoid inplace change of input x_raw
        x = self.prepare_input(x_raw.clone())
        shape = x.shape[:-1]
        x = x.view(-1, x.size(-1)).to(self.dtype)
        if l is None:
            l = torch.zeros_like(x[:,[0]]).to(self.dtype)
        else:
            l = l.view(-1, 1).to(self.dtype)
        # Q can reside in higher Brillouin zones
        # Q = torch.cat((x[:,:2], torch.zeros_like(x[:,1])[:,None]), dim=1)
        # Reduced reciprocal lattice vectors projected into the first quadrant of the Brillouin zone
        # since the models are trained on the first quadrant only
        Q = torch.cat((x[:,:2], l), dim=1)
        x[:,:2] = torch.abs(x[:,:2] - torch.round(x[:,:2]))
        if Syy is None:
            Syy = self.Syy_net(x).squeeze(-1)
        else:
            Syy = Syy.view(-1).to(self.dtype)
        if Szz is None:
            Szz = self.Szz_net(x).squeeze(-1)
        else:
            Szz = Szz.view(-1).to(self.dtype)
        S = self.calculate_Sqw(Q, Syy, Szz)
        return S.view(shape)
        
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value.view(2).to(self.dtype).to(self.device)
    
    @torch.no_grad()
    def forward_qw(self, x_hklw, params=None):
        x_hklw = x_hklw.to(self.dtype).to(self.device)
        if params is None:
            params = self.params.to(self.dtype).to(self.device)
        else:
            params = params.view(2).to(self.dtype).to(self.device)
        _x_hkw = x_hklw[...,[0,1,3]].reshape(-1, 3).contiguous()
        x_raw = torch.cat([_x_hkw, params.unsqueeze(0).repeat(_x_hkw.size(0), 1)], dim=1)
        output = self.forward(x_raw, x_hklw[...,[2]].reshape(-1, 1))
        return output.reshape(x_hklw.shape[:-1]).contiguous()
    
    def forward_qwp(self, x):
        """x of shape (..., 6), each component in the last dimension is: (h, k, l, w, J, Jp)
        """
        x_raw, l = x[...,[0,1,3,4,5]], x[...,[2]]
        return self.forward(x_raw, l)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """Assuming reciprocal lattice vectors are in the first quadrant of the Brillouin zone
           Thus no need to project them into the first quadrant
        """
        x, (Syy, Szz) = train_batch
        x = self.prepare_input(x)
        x = x.view(-1, x.size(-1)).to(self.dtype)
        Syy = Syy.view(-1, Syy.size(-1)).to(self.dtype)
        Szz = Szz.view(-1, Szz.size(-1)).to(self.dtype)
        
        Syy_pred = self.Syy_net(x)
        Szz_pred = self.Szz_net(x)
        
        loss_Syy = F.mse_loss(Syy_pred, Syy)
        loss_Szz = F.mse_loss(Szz_pred, Szz)
        loss = loss_Syy + loss_Szz
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, (Syy, Szz) = val_batch
        x = self.prepare_input(x)
        x = x.view(-1, x.size(-1)).to(self.dtype)
        Syy = Syy.view(-1, Syy.size(-1)).to(self.dtype)
        Szz = Szz.view(-1, Szz.size(-1)).to(self.dtype)
        
        Syy_pred = self.Syy_net(x)
        Szz_pred = self.Szz_net(x)
        
        loss_Syy = F.mse_loss(Syy_pred, Syy)
        loss_Szz = F.mse_loss(Szz_pred, Szz)
        loss = loss_Syy + loss_Szz
        self.log('val_loss', loss)

    def formfact(self, Q):
        H, K, L = Q[:,0], Q[:,1], Q[:,2]
        a, c = self.latt_const.a, self.latt_const.c
        
        q = 2 * np.pi * torch.sqrt((H**2 + K**2) / a**2 + L**2 / c**2) # Scattering vector in Angstroem^-1
        s2 = (q/4/np.pi)**2
        j0 = (self.ff.A0 * torch.exp(-self.ff.a0*s2) + 
              self.ff.B0 * torch.exp(-self.ff.b0*s2) + 
              self.ff.C0 * torch.exp(-self.ff.c0*s2) + self.ff.D0)
        j4 = (self.ff.A4 * torch.exp(-self.ff.a4*s2) + 
              self.ff.B4 * torch.exp(-self.ff.b4*s2) + 
              self.ff.C4 * torch.exp(-self.ff.c4*s2) + self.ff.D4) * s2
        ff_q = j0 + j4 * 3/2 * (
                H**4 + K**4 + L**4 * (a/c)**4 - 
                3 * (H**2 * K**2 + H**2 * L**2 * (a/c)**2 + K**2 * L**2 * (a/c)**2)
            ) / (H**2 + K**2 + L**2 * (a/c)**2 + 1e-15) ** 2
        return ff_q
    
    def calculate_Sqw(self, Q, Syy, Szz):
        H, K, L = Q[:,0], Q[:,1], Q[:,2]
        a, c = self.latt_const.a, self.latt_const.c
        ql = 2 * np.pi * L / c # Out of plane component of the scattering vector
        q = 2 * np.pi * torch.sqrt((H**2 + K**2) / a**2 + L**2 / c**2) # Scattering vector in Angstroem^-1
        
        ff_q = self.formfact(Q).detach()
        S = (torch.abs(ff_q)**2) * (
            (1 + (ql/(q+1e-15))**2) / 2 * Syy + (1 - (ql/(q+1e-15))**2) * Szz
        )
        return S