import torch
import torch.nn as nn
import torch.nn.functional as F




class time_latent_module(torch.nn.Module):
    def __init__(self,T,D,max_norm=1.0):
        super().__init__()
        self.time_emb = nn.Embedding(T, D, max_norm=max_norm) 
        self.T = T
        self.D = D
        

    def forward(self, t):  #t, deg):
        num_frames = self.T
        time = (t + 1) / 2 * (num_frames - 1)
        t0 = int(torch.floor(time).item())
        t1 = min(t0 + 1, num_frames - 1)  # Clamp to valid index
        alpha = time - t0
        t0_tensor = torch.Tensor([t0]).long().to('cuda:0')
        t1_tensor = torch.Tensor([t1]).long().to('cuda:0')
        #print(t0, t1)
        return torch.lerp(self.time_emb(t0_tensor), self.time_emb(t1_tensor), alpha)
    


class time_latent_hierachy_module(torch.nn.Module):
    def __init__(self,T,D,max_norm=1.0):
        super().__init__()
        self.time_emb = nn.Embedding(T, D, max_norm=max_norm) 
        self.T = T
        self.D = D
        self.lv1 = time_latent_module(T//8,D) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        self.lv2 = time_latent_module(T//4,D) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        self.lv3 = time_latent_module(T//2,D) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        self.lv4 = time_latent_module(T,D) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)
        

    def forward(self, t):  #t, deg):
        sampled_lv1 = self.lv1(t).squeeze()
        sampled_lv2 = self.lv2(t).squeeze()
        sampled_lv3 = self.lv3(t).squeeze()
        sampled_lv4 = self.lv4(t).squeeze()
        #print(sampled_lv1.shape, sampled_lv2.shape, sampled_lv3.shape, sampled_lv4.shape)
        sampled_times = (sampled_lv1+sampled_lv2+sampled_lv3+sampled_lv4)/4
        return sampled_times
        
class time_latent_module_unnorm(torch.nn.Module):
    def __init__(self,T,D,max_norm=1.0):
        super().__init__()
        self.time_emb = torch.nn.parameter.Parameter(torch.rand(T, D))
        #nn.Embedding(T, D, max_norm=max_norm) 
        self.T = T
        self.D = D
        

    def forward(self, t):  #t, deg):
        num_frames = self.T
        time = (t + 1) / 2 * (num_frames - 1)
        t0 = int(torch.floor(time).item())
        t1 = min(t0 + 1, num_frames - 1)  # Clamp to valid index
        alpha = time - t0
        t0_tensor = t0 #torch.Tensor([t0]).long().to('cuda:0')
        t1_tensor = t1 #torch.Tensor([t1]).long().to('cuda:0')
        #print(t0, t1)
        return torch.lerp(self.time_emb[t0_tensor], self.time_emb[t1_tensor], alpha)

        
class time_latent_hierachy_module_unnorm(torch.nn.Module):
    def __init__(self,T,D,max_norm=1.0):
        super().__init__()
        self.T = T
        self.D = D
        self.lv1 = time_latent_module_unnorm(T//8,D) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        self.lv2 = time_latent_module_unnorm(T//4,D) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        self.lv3 = time_latent_module_unnorm(T//2,D) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        self.lv4 = time_latent_module_unnorm(T,D) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)
        
    def forward(self, t):  #t, deg):
        sampled_lv1 = self.lv1(t).squeeze()
        sampled_lv2 = self.lv2(t).squeeze()
        sampled_lv3 = self.lv3(t).squeeze()
        sampled_lv4 = self.lv4(t).squeeze()
        #print(sampled_lv1.shape, sampled_lv2.shape, sampled_lv3.shape, sampled_lv4.shape)
        sampled_times = (sampled_lv1+sampled_lv2+sampled_lv3+sampled_lv4)/4
        return sampled_times

max_norm=1.0

class model_gs_deg_time_view_hierachy_new(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        # self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        self.lv1 = time_latent_module(t//8,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        self.lv2 = time_latent_module(t//4,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        self.lv3 = time_latent_module(t//2,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        self.lv4 = time_latent_module(t,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        self.num_v = num_v
        self.lt = lt
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
        #sampled_times = F.tanh(self.time_embedding(t).squeeze())
        # pos_idx = torch.zeros((1,1,1,2)).to('cuda:0')
        # pos_idx[0][0][0][0] = t
        # sampled_lv1 = F.grid_sample(self.lv1, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        # sampled_lv2 = F.grid_sample(self.lv2, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        # sampled_lv3 = F.grid_sample(self.lv3, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        # sampled_lv4 = F.grid_sample(self.lv4, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_lv1 = self.lv1(t_tensor).squeeze()
        sampled_lv2 = self.lv2(t_tensor).squeeze()
        sampled_lv3 = self.lv3(t_tensor).squeeze()
        sampled_lv4 = self.lv4(t_tensor).squeeze()
        #print(sampled_lv1.shape, sampled_lv2.shape, sampled_lv3.shape, sampled_lv4.shape)
        sampled_times = F.tanh((sampled_lv1+sampled_lv2+sampled_lv3+sampled_lv4)/4).unsqueeze(0).expand((self.num_v,self.lt))

        #sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        #x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2
    

class model_gs_deg_time_view_hierachy_new_unnorm(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        # self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        # self.lv1 = time_latent_module(t//8,lt,max_norm=None) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv2 = time_latent_module(t//4,lt,max_norm=None)  #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        # self.lv3 = time_latent_module(t//2,lt,max_norm=None)  #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = time_latent_module(t,lt,max_norm=None)  #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)
        self.time_module = time_latent_hierachy_module(t, lt, max_norm=None)
        self.num_v = num_v
        self.lt = lt
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
        #sampled_times = F.tanh(self.time_embedding(t).squeeze())
        # pos_idx = torch.zeros((1,1,1,2)).to('cuda:0')
        # pos_idx[0][0][0][0] = t
        # sampled_lv1 = F.grid_sample(self.lv1, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        # sampled_lv2 = F.grid_sample(self.lv2, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        # sampled_lv3 = F.grid_sample(self.lv3, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        # sampled_lv4 = F.grid_sample(self.lv4, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        t_tensor = torch.Tensor([t]).to('cuda:0')
        # sampled_lv1 = self.lv1(t_tensor).squeeze()
        # sampled_lv2 = self.lv2(t_tensor).squeeze()
        # sampled_lv3 = self.lv3(t_tensor).squeeze()
        # sampled_lv4 = self.lv4(t_tensor).squeeze()
        sampled_time = self.time_module(t_tensor)
        #print(sampled_lv1.shape, sampled_lv2.shape, sampled_lv3.shape, sampled_lv4.shape)
        sampled_times = F.tanh(sampled_time).unsqueeze(0).expand((self.num_v,self.lt))

        #sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        #x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2
    



class model_gs_deg_time_view_hierachy_new_viewdep(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        #self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        # self.lv1 = time_latent_module(t//8,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv2 = time_latent_module(t//4,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        # self.lv3 = time_latent_module(t//2,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = time_latent_module(t,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_times = self.time_modules[deg](t_tensor)
        sampled_times = F.tanh(sampled_times).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_times], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2


class model_gs_deg_time_view_hierachy_new_viewdep_share_time(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        #self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        # self.lv1 = time_latent_module(t//8,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv2 = time_latent_module(t//4,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        # self.lv3 = time_latent_module(t//2,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = time_latent_module(t,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)
        self.time_module = time_latent_hierachy_module(t, lt)
        self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_time = self.time_module(t_tensor)
        sampled_times = self.time_modules[deg](t_tensor)
        sampled_times = F.tanh(sampled_time + sampled_times).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_times], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2

class model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        #self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        # self.lv1 = time_latent_module(t//8,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv2 = time_latent_module(t//4,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        # self.lv3 = time_latent_module(t//2,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = time_latent_module(t,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)
        self.time_module = time_latent_hierachy_module(t, lt)
        self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_time = self.time_module(t_tensor)
        sampled_times = self.time_modules[deg](t_tensor)

        sampled_tt = F.tanh(sampled_time).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_view = F.tanh(sampled_time + sampled_times).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_tt], 1)
        x_view = torch.cat([sampled_positions,sampled_view], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_view)
        d_rots = self.model_r(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_view)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2



class model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo_stall_latent(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        #self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        # self.lv1 = time_latent_module(t//8,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv2 = time_latent_module(t//4,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        # self.lv3 = time_latent_module(t//2,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = time_latent_module(t,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)
        self.time_module = nn.Embedding(t, lt, max_norm=1.0) #time_latent_hierachy_module(t, lt)
        # self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module(t, lt) for i in range(deg_s)])

        self.time_modules = torch.nn.ModuleList([nn.Embedding(t, lt, max_norm=1.0) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        t_tensor = torch.Tensor([t]).long().squeeze().to('cuda:0')
        # sampled_time = self.time_module(t_tensor)
        #print(deg, t_tensor)
        sampled_times = self.time_modules[deg](t_tensor)

        sampled_tt = F.tanh(sampled_times).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_view = F.tanh(sampled_times).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_tt], 1)
        x_view = torch.cat([sampled_positions,sampled_view], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_view)
        d_rots = self.model_r(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_view)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2

        

class model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo_unnorm1(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_module = time_latent_hierachy_module(t, lt)
        self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_time = self.time_module(t_tensor)
        sampled_times = self.time_modules[deg](t_tensor)

        sampled_tt = (sampled_time).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_view = (sampled_time + sampled_times).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_tt], 1)
        x_view = torch.cat([sampled_positions,sampled_view], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_view)
        d_rots = self.model_r(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_view)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws

        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2

class model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo_amplify(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538,amplify=1.0):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.amplify = amplify
        #self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        # self.lv1 = time_latent_module(t//8,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv2 = time_latent_module(t//4,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        # self.lv3 = time_latent_module(t//2,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = time_latent_module(t,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)
        self.time_module = time_latent_hierachy_module(t, lt)
        self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_time = self.time_module(t_tensor)
        sampled_times = self.time_modules[deg](t_tensor)

        sampled_tt = (F.tanh(sampled_time)*self.amplify).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_view = (F.tanh(sampled_time + sampled_times)*self.amplify).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_tt], 1)
        x_view = torch.cat([sampled_positions,sampled_view], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_view)
        d_rots = self.model_r(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_view)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2

class model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo_unnorm2(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 

        self.time_module = time_latent_hierachy_module_unnorm(t, lt)
        self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module_unnorm(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_time = self.time_module(t_tensor)
        sampled_times = self.time_modules[deg](t_tensor)

        sampled_tt = F.tanh(sampled_time).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_view = F.tanh(sampled_time + sampled_times).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_tt], 1)
        x_view = torch.cat([sampled_positions,sampled_view], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_view)
        d_rots = self.model_r(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_view)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2
    

class model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo_unnorm_noview(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 

        self.time_module = time_latent_hierachy_module_unnorm(t, lt)
        self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module_unnorm(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_time = self.time_module(t_tensor)

        sampled_tt = F.tanh(sampled_time).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_view = F.tanh(sampled_time).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_tt], 1)
        x_view = torch.cat([sampled_positions,sampled_view], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_view)
        d_rots = self.model_r(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_view)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2


class model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo_unnorm_noview_normlast(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 

        self.time_module = time_latent_hierachy_module_unnorm(t, lt)
        self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module_unnorm(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_time = self.time_module(t_tensor)

        sampled_tt = torch.nn.functional.normalize(sampled_time, dim=0).unsqueeze(0).expand((self.num_v,self.lt)) #F.tanh(sampled_time).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_view = torch.nn.functional.normalize(sampled_time, dim=0).unsqueeze(0).expand((self.num_v,self.lt)) #F.tanh(sampled_time).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_tt], 1)
        x_view = torch.cat([sampled_positions,sampled_view], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_view)
        d_rots = self.model_r(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_view)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2



class model_gs_deg_time_view_hierachy_new_triplane_h_pos_lowres_lowtime_triplane_level(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538,gres=64):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=None) 
        nn.init.zeros_(self.pos_embedding.weight)
        # self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        self.grid_xy = torch.nn.parameter.Parameter(data=torch.rand(1,64,8,8), requires_grad=True)
        self.grid_xz = torch.nn.parameter.Parameter(data=torch.rand(1,64,8,8), requires_grad=True)
        self.grid_yz = torch.nn.parameter.Parameter(data=torch.rand(1,64,8,8), requires_grad=True)

        # self.grid_xy_16 = torch.nn.parameter.Parameter(data=torch.rand(1,64,16,16)*0.01, requires_grad=True)
        # self.grid_xz_16 = torch.nn.parameter.Parameter(data=torch.rand(1,64,16,16)*0.01, requires_grad=True)
        # self.grid_yz_16 = torch.nn.parameter.Parameter(data=torch.rand(1,64,16,16)*0.01, requires_grad=True)

        # self.grid_xy_32 = torch.nn.parameter.Parameter(data=torch.rand(1,64,32,32)*0.01, requires_grad=True)
        # self.grid_xz_32 = torch.nn.parameter.Parameter(data=torch.rand(1,64,32,32)*0.01, requires_grad=True)
        # self.grid_yz_32 = torch.nn.parameter.Parameter(data=torch.rand(1,64,32,32)*0.01, requires_grad=True)

        # self.grid_xy_64 = torch.nn.parameter.Parameter(data=torch.rand(1,64,64,64)*0.01, requires_grad=True)
        # self.grid_xz_64 = torch.nn.parameter.Parameter(data=torch.rand(1,64,64,64)*0.01, requires_grad=True)
        # self.grid_yz_64 = torch.nn.parameter.Parameter(data=torch.rand(1,64,64,64)*0.01, requires_grad=True)

        self.lv1 = time_latent_module(t//8,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        self.lv2 = time_latent_module(t//4,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        self.lv3 = time_latent_module(t//2,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        self.lv4 = time_latent_module(t,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        self.num_v = num_v
        self.lt = lt
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
        

    def forward(self, sample_coord, t, deg):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        # sampled_positions_perlatent = self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0'))
        sample_xy = torch.nn.functional.grid_sample(self.grid_xy, sample_coord[:,:,:,[0,1]])
        sample_xz = torch.nn.functional.grid_sample(self.grid_xz, sample_coord[:,:,:,[0,2]])
        sample_yz = torch.nn.functional.grid_sample(self.grid_yz, sample_coord[:,:,:,[1,2]])
        sample_combine = sample_xy*sample_xz*sample_yz
        #sample_combine_8 = F.normalize(sample_combine_8.squeeze().T, dim=1)
        #print(sample_combine_8.shape)
        sampled_positions = sample_combine.squeeze().T #sampled_positions_perlatent #sample_combine_32 #+sample_combine_16+sample_combine_32+sampled_positions_perlatent

        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_lv1 = self.lv1(t_tensor).squeeze()
        sampled_lv2 = self.lv2(t_tensor).squeeze()
        sampled_lv3 = self.lv3(t_tensor).squeeze()
        sampled_lv4 = self.lv4(t_tensor).squeeze()
        #print(sampled_lv1.shape, sampled_lv2.shape, sampled_lv3.shape, sampled_lv4.shape)
        #sampled_times = (sampled_lv1).unsqueeze(0).expand((self.num_v,self.lt))
        #(sampled_lv1+sampled_lv2+sampled_lv3+sampled_lv4).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_times = (sampled_lv1+sampled_lv2+sampled_lv3+sampled_lv4).unsqueeze(0).expand((self.num_v,self.lt))

        #sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        #print(sampled_positions.shape, sampled_times.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        #x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)


        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2
        
    def upsample_latent(self):
        self.grid_xy = torch.nn.parameter.Parameter(data=F.interpolate(self.grid_xy,scale_factor=2, mode='bilinear',align_corners=False),requires_grad=True)
        self.grid_xz = torch.nn.parameter.Parameter(data=F.interpolate(self.grid_xz,scale_factor=2, mode='bilinear',align_corners=False),requires_grad=True)
        self.grid_yz = torch.nn.parameter.Parameter(data=F.interpolate(self.grid_yz,scale_factor=2, mode='bilinear',align_corners=False),requires_grad=True)


class model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo_detach(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_module = time_latent_hierachy_module(t, lt)
        self.time_modules = torch.nn.ModuleList([time_latent_hierachy_module(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        

    def forward(self, t, deg, is_detach):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_time = self.time_module(t_tensor)
        if is_detach:
            sampled_times = self.time_modules[deg](t_tensor).detach()
        else:
            sampled_times = self.time_modules[deg](t_tensor)

        sampled_tt = F.tanh(sampled_time).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_view = F.tanh(sampled_time + sampled_times).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_tt], 1)
        x_view = torch.cat([sampled_positions,sampled_view], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_view)
        d_rots = self.model_r(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_view)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2

class model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo_sparseview(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_module = time_latent_hierachy_module(t, lt)
        self.time_modules = nn.parameter.Parameter(torch.rand(deg_s,t,lt))
        #torch.nn.ModuleList([time_latent_hierachy_module(t, lt) for i in range(deg_s)])

        self.num_v = num_v
        self.lt = lt
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
        

    def forward(self, t, deg, t_idx):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_time = self.time_module(t_tensor)
        sampled_times = self.time_modules[deg][t_idx]

        sampled_tt = F.tanh(sampled_time).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_view = F.tanh(sampled_time + sampled_times).unsqueeze(0).expand((self.num_v,self.lt))

        x_time = torch.cat([sampled_positions,sampled_tt], 1)
        x_view = torch.cat([sampled_positions,sampled_view], 1)

        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_view)
        d_rots = self.model_r(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_view)

        d_alphas2 = d_alphas
        d_scales2 = d_scales
        d_rots2 = d_rots
        d_low_shs2 = d_low_shs
        d_pws2 = d_pws
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2
    


class model_gs_deg_time_view_hierachy_tensorcompression(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538,coeff_num=100):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.coeff_num = coeff_num
        #self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        self.lv1 = time_latent_module(t//8,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        self.lv2 = time_latent_module(t//4,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        self.lv3 = time_latent_module(t//2,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        self.lv4 = time_latent_module(t,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        self.coeff = nn.Parameter(torch.randn(t, deg_s, coeff_num)*0.001)
        self.basis = nn.Parameter(torch.randn(coeff_num, num_v, 10)*0.001)

        self.num_v = num_v
        self.lt = lt
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
            nn.Linear(d, 3),
        )
        

    def forward(self, t,t_index, deg):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        #sampled_times = F.tanh(self.time_embedding(t).squeeze())
        # pos_idx = torch.zeros((1,1,1,2)).to('cuda:0')
        # pos_idx[0][0][0][0] = t
        # sampled_lv1 = F.grid_sample(self.lv1, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        # sampled_lv2 = F.grid_sample(self.lv2, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        # sampled_lv3 = F.grid_sample(self.lv3, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        # sampled_lv4 = F.grid_sample(self.lv4, pos_idx, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze()
        t_tensor = torch.Tensor([t]).to('cuda:0')
        sampled_lv1 = self.lv1(t_tensor).squeeze()
        sampled_lv2 = self.lv2(t_tensor).squeeze()
        sampled_lv3 = self.lv3(t_tensor).squeeze()
        sampled_lv4 = self.lv4(t_tensor).squeeze()
        #print(sampled_lv1.shape, sampled_lv2.shape, sampled_lv3.shape, sampled_lv4.shape)
        sampled_times = F.tanh((sampled_lv1+sampled_lv2+sampled_lv3+sampled_lv4)/4).unsqueeze(0).expand((self.num_v,self.lt))

        #sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        #x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)



        d_alphas = self.model_a(x_time)
        d_scales = self.model_s(x_time)
        d_rots = self.model_r(x_time)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x(x_time)

        # time = (t + 1) / 2 * (num_frames - 1)
        # t0 = int(torch.floor(time).item())
        # t1 = min(t0 + 1, num_frames - 1) 
        # print(t_index)
        # print(deg)
        # print(self.coeff.shape,self.coeff[t_index][deg].shape,  self.basis.shape)
        d_offset = torch.einsum('j,jkl->kl', self.coeff[t_index][deg], self.basis) #torch.matmul(self.coeff[t_index][deg], self.basis).squeeze()
        d_alphas2 = d_alphas
        d_scales2 = d_offset[:,:3]
        d_rots2 = d_offset[:,3:7]
        d_low_shs2 = d_low_shs
        d_pws2 = d_offset[:,7:]
        # d_alphas2 = self.model_a2(x_view)
        # d_scales2 = self.model_s2(x_view)
        # d_rots2 = self.model_r2(x_view)
        # d_low_shs2 = self.model_c2(x_view)
        # d_pws2 = self.model_x2(x_view)
        
        return d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2