import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from layers.Embed import DataEmbedding

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        self.d_inner = configs.d_model * configs.expand
        self.dt_rank = math.ceil(configs.d_model / 16) # TODO implement "auto"
        
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Decomposition Modules
        self.ReLU = nn.ReLU()
        self.fc0 = torch.nn.Linear(configs.enc_in, configs.c_out, bias=True)
        self.fc1 = torch.nn.Linear(self.pred_len, 128, bias=True)
        self.fc2 = torch.nn.Linear(128, 64, bias=True)
        self.fc3 = torch.nn.Linear(64, configs.c_out, bias=True)

        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )

        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc

        dec_out = self.ReLU(x_out)

        x = self.fc0(dec_out)
        x = self.ReLU(x)
        
        x = self.fc1(x.squeeze(-1))
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x_out = self.fc3(x)

        #print(x_out.shape, "is the shape of x_out")
        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out
            #return x_out[:, -self.pred_len:, :]

        # other tasks not implemented