import torch
import torch.nn as nn
import torch.nn.functional as F




class time_latent_module(torch.nn.Module):
    def __init__(self,T,D):
        super().__init__()
        self.time_emb = nn.Embedding(T, D, max_norm=1.0) 
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

class hierachy_time_module(torch.nn.Module):
    def __init__(self,t,lt):
        super().__init__()
        self.lv1 = time_latent_module(t//8,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        self.lv2 = time_latent_module(t//4,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
        self.lv3 = time_latent_module(t//2,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        self.lv4 = time_latent_module(t,lt)
        

    def forward(self, t_tensor):  #t, deg):
        sampled_lv1 = self.lv1(t_tensor).squeeze()
        sampled_lv2 = self.lv2(t_tensor).squeeze()
        sampled_lv3 = self.lv3(t_tensor).squeeze()
        sampled_lv4 = self.lv4(t_tensor).squeeze()
        #print(t0, t1)
        return (sampled_lv1+sampled_lv2+sampled_lv3+sampled_lv4)/4
    


class model_gs_deg_time_view_hierachy_new_vewdep(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        #self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
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

        sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)



        d_alphas = self.model_a(x_time)
        d_scales = self.model_s2(x_view)
        d_rots = self.model_r2(x_view)
        d_low_shs = self.model_c(x_time)
        d_pws = self.model_x2(x_view)

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



class model_gs_deg_time_view_hierachy_new_vewdep_hierachy(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        #self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        # self.lv1 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//8), requires_grad=True)
        # self.lv2 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        # self.lv3 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//2), requires_grad=True)
        # self.lv4 = torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t), requires_grad=True)

        self.h_time = hierachy_time_module(t,lt) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//4), requires_grad=True)
        self.h_view = torch.nn.ModuleList([time_latent_module(t, lt) for i in range(deg_s)]) #torch.nn.parameter.Parameter(data=torch.rand(1,lt,1,t//3), requires_grad=True)
      
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
        #print(sampled_lv1.shape, sampled_lv2.shape, sampled_lv3.shape, sampled_lv4.shape)
        t_latent = self.h_time(t_tensor)
        #print(deg[0][0][0])
        v_latent = self.h_view[deg[0][0][0].item()](t_tensor).squeeze()

        #print(t_latent.shape, v_latent.shape)
        sampled_times = F.tanh(t_latent).unsqueeze(0).expand((self.num_v,self.lt))
        sampled_degs = F.tanh(t_latent+v_latent).unsqueeze(0).expand((self.num_v,self.lt))
        #sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        x_view = torch.cat([sampled_positions,sampled_degs], 1)


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