import torch
import torch.nn as nn
import torch.nn.functional as F




class triplane_hashgrid(torch.nn.Module):
    def __init__(self,dim=64,dimout=64,sz=1024):
        super().__init__()
        self.xy = torch.nn.Parameter(torch.randn((1,dim,sz,sz)))
        self.xz = torch.nn.Parameter(torch.randn((1,dim,sz,sz)))
        self.yz = torch.nn.Parameter(torch.randn((1,dim,sz,sz)))
        self.lin = nn.Linear(dim*3,dimout)

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        xy_tensor = F.grid_sample(self.xy,x[:,:,:,[0,1]])
        xz_tensor = F.grid_sample(self.xz,x[:,:,:,[0,2]])
        yz_tensor = F.grid_sample(self.yz,x[:,:,:,[1,2]])
        xyz_tensor = torch.cat([xy_tensor,xz_tensor,yz_tensor],1).squeeze().T
        #print(xyz_tensor.shape)
        x = self.lin(xyz_tensor) #.squeeze()
        return x

class hierachical_triplane(torch.nn.Module):
    def __init__(self,dim=64,dimout=64,level = [256,512,1024]):
        super().__init__()
        self.h0 = triplane_hashgrid(dim,dimout,level[0])
        self.h1 = triplane_hashgrid(dim,dimout,level[1])
        self.h2 = triplane_hashgrid(dim,dimout,level[2])


    def forward(self, x):
        h0 = self.h0(x)
        h1 = self.h1(x)
        h2 = self.h2(x)
        hs = h0+h1+h2 #torch.cat([h0,h1,h2], 1)
        return hs


class time_coefficient_compression(torch.nn.Module):
    def __init__(self,num_time=500,coefficient_dim=256,codebooksize=50):
        super().__init__()
        self.emb = nn.Embedding(num_time,codebooksize)
        self.codebooks = nn.Parameter(torch.rand((codebooksize, coefficient_dim)))
        self.coefficient_dim = coefficient_dim
        self.num_time = num_time
        self.codebooksize = codebooksize

    def forward(self, x):
        coeff = self.emb(x)
        #print(coeff.shape)
        latent = torch.matmul(coeff,self.codebooks) #.expand(x.shape[0],self.coefficient_dim ) #.repeat((x.shape[0],self.coefficient_dim ))
        #print(latent.shape)
        return latent



# a = hierachical_triplane().to('cuda:0')
# b = torch.randn(1000,3).to('cuda:0')*2-1
# c = a(b)
# print(c.shape)
# tt = time_coefficient_compression().to('cuda:0')
# t = torch.ones(1000).long().to('cuda:0')
# t_out = tt(t)

# concat_input = torch.cat([c, t_out], 1)
# print(concat_input.shape)

class model_gs_deg_time_view_compress(torch.nn.Module):
    def __init__(self,sz=128,lt=64,ltt=64,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = hierachical_triplane() #nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_embedding = time_coefficient_compression(coefficient_dim=lt) #nn.Embedding(t, lt, max_norm=1.0)
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
        

    def forward(self, x, t, deg):  #t, deg):
        
        sampled_positions =  self.pos_embedding(x).squeeze() #F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        sampled_times = self.time_embedding(t).squeeze() #F.tanh(self.time_embedding(t).squeeze())
        sampled_degs = self.deg_embedding(deg).squeeze() #F.tanh(self.deg_embedding(deg).squeeze())

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
        self.pos_embedding = hierachical_triplane(level = [256,512,1024])
        #nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)  #time_coefficient_compression(coefficient_dim=lt) #nn.Embedding(t, lt, max_norm=1.0)
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

# mm = model_gs_deg_time_view_compress().to('cuda:0')
# x = torch.randn(1000,3).to('cuda:0')*2-1
# t = torch.ones(1000).long().to('cuda:0')
# t2 = torch.ones(1000).long().to('cuda:0')

# d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2 = mm(x,t,t2)


# print(d_pws.shape)