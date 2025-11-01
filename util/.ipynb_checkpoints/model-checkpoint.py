import torch
import torch.nn as nn
import torch.nn.functional as F

class model_gs(torch.nn.Module):
    def __init__(self,sz=128,lt=128,ltt=128,d=256,t=297):
        super().__init__()
        self.pos_embedding = nn.Embedding(50538, 64, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, 256, max_norm=1.0)
        self.deg_embedding = nn.Embedding(t, 64, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(64+256, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(64+256, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(64+256, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(64+256, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(64+256, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self,t ): #, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        x = torch.cat([sampled_positions,sampled_times], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs



class model_gs_deg(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, 64, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(deg_s, ltt, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, t, deg):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs


    def forward_latent(self, sampled_times, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_times = F.tanh(sampled_times.squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())
        print(sampled_times.shape, sampled_degs.shape)
        x = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs
    


class model_gs_deg_time_view_possitional_embedding(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        #self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(deg_s, ltt, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )

        self.model_x2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, t, deg):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions =  F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        sampled_times = t #F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)
        
        d_alphas2 = self.model_a2(x_view)
        d_scales2 = self.model_s2(x_view)
        d_rots2 = self.model_r2(x_view)
        d_low_shs2 = self.model_c2(x_view)
        d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2


class model_gs_deg_time_view(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(deg_s, ltt, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )

        self.model_x2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, t, deg):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)
        
        d_alphas2 = self.model_a2(x_view)
        d_scales2 = self.model_s2(x_view)
        d_rots2 = self.model_r2(x_view)
        d_low_shs2 = self.model_c2(x_view)
        d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2


class model_gs_deg_time_view2(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(deg_s, ltt, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )

        self.model_x2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, t, deg):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)
        
        d_alphas2 = self.model_a2(x_view)
        d_scales2 = self.model_s2(x_view)
        d_rots2 = self.model_r2(x_view)
        d_low_shs2 = self.model_c2(x_view)
        d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2
    

class model_gs_deg_time_view_6layer(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(deg_s, ltt, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )

        self.model_x2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, t, deg):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)
        
        d_alphas2 = self.model_a2(x_view)
        d_scales2 = self.model_s2(x_view)
        d_rots2 = self.model_r2(x_view)
        d_low_shs2 = self.model_c2(x_view)
        d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2


class model_gs_deg_time_view_fast(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(deg_s, ltt, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )

        self.model_x2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c2 = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, pos, t, deg):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(pos))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)
        
        d_alphas2 = self.model_a2(x_view)
        d_scales2 = self.model_s2(x_view)
        d_rots2 = self.model_r2(x_view)
        d_low_shs2 = self.model_c2(x_view)
        d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2



class model_gs_deg_time_view_share(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(deg_s, ltt, max_norm=1.0)
        
        self.model_time = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3+1+4+3+3),
        )

        self.model_view = nn.Sequential(
            nn.Linear(per_gs_lt+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3+1+4+3+3),
        )
        

    def forward(self, t, deg):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)

        d_all = self.model_time(x_time)
        d_all2 = self.model_view(x_view)

        d_pws = d_all[:,[0,1,2]]
        d_alphas = d_all[:,[3]]
        d_scales = d_all[:,[4,5,6]]
        d_rots = d_all[:,[7,8,9,10]]
        d_low_shs = d_all[:,[11,12,13]]

        d_pws2 = d_all2[:,[0,1,2]]
        d_alphas2 = d_all2[:,[3]]
        d_scales2 = d_all2[:,[4,5,6]]
        d_rots2 = d_all2[:,[7,8,9,10]]
        d_low_shs2 = d_all2[:,[11,12,13]]
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2





class model_gs_deg_retopo(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,d=256,t=297):
        super().__init__()
        self.pos_embedding = nn.Embedding(150615, 64, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(30, ltt, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, t, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(150615).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())
        x = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs


    def forward_latent(self, sampled_times, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(150615).to('cuda:0')))
        sampled_times = F.tanh(sampled_times.squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())
        print(sampled_times.shape, sampled_degs.shape)
        x = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs

class model_gs_deg_mulan(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,d=256,t=297):
        super().__init__()
        self.pos_embedding = nn.Embedding(1823, 64, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(30, ltt, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, t, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(1823).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())
        x = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs


    def forward_latent(self, sampled_times, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_times = F.tanh(sampled_times.squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())
        print(sampled_times.shape, sampled_degs.shape)
        x = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs


class model_gs_deg_vary(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,d=256,t=297):
        super().__init__()
        self.pos_embedding = nn.Embedding(50538, 64, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding_d = nn.Embedding(30, 16, max_norm=1.0)
        self.deg_embedding_c = nn.Embedding(30, 256, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(64+lt+16, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(64+lt+16, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(64+lt+16, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(64+lt+16, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(64+lt+256, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, t, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs_d = F.tanh(self.deg_embedding_d(deg).squeeze())
        sampled_degs_c = F.tanh(self.deg_embedding_c(deg).squeeze())

        x1 = torch.cat([sampled_positions,sampled_times,sampled_degs_d], 1)
        x2 = torch.cat([sampled_positions,sampled_times,sampled_degs_c], 1)


        d_alphas = self.model_a(x1) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x1) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x1) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x2) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x1) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs


    def forward_latent(self, sampled_times, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_times = F.tanh(sampled_times.squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())
        print(sampled_times.shape, sampled_degs.shape)
        x = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs


class model_gs_deg_detach(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,d=256,t=297):
        super().__init__()
        self.pos_embedding = nn.Embedding(50538, 64, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.deg_embedding = nn.Embedding(30, ltt, max_norm=1.0)
        
        self.model_x = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_a = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.model_r = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )
        self.model_s = nn.Sequential(
            nn.Linear(64+lt+ltt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.model_c = nn.Sequential(
            nn.Linear(64+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        

    def forward(self, t, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_times = F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())
        x = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)
        x2 = torch.cat([sampled_positions,sampled_times], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x2) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs


    def forward_latent(self, sampled_times, deg):  #t, deg):
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_times = F.tanh(sampled_times.squeeze())
        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())
        print(sampled_times.shape, sampled_degs.shape)
        x = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x) #x[:,:,[0]].squeeze(0)
        d_scales = self.model_s(x) #x[:,:,1:4].squeeze(0)
        d_rots = self.model_r(x) #x[:,:,4:8].squeeze(0)
        d_low_shs = self.model_c(x) #x[:,:,8:11].squeeze(0)
        d_pws = self.model_x(x) #x[:,:,11:].squeeze(0)
        # d_shs = get_shs(low_shs, high_shs)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs


