import torch
import torch.nn as nn
import torch.nn.functional as F


class model_gs_deg_time_view_view_emb(torch.nn.Module):
    def __init__(self,sz=128,lt=256,ltt=256,per_gs_lt=64,d=256,t=297,deg_s=30,num_v=50538):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_v, per_gs_lt, max_norm=1.0) 
        self.time_embedding = nn.Embedding(t, lt, max_norm=1.0)
        self.view_embedding = torch.nn.ModuleList([nn.Embedding(t, lt, max_norm=1.0) for i in range(20)])


        #self.deg_embedding = nn.Embedding(deg_s, ltt, max_norm=1.0)

        self.model_offset = nn.Sequential(
            nn.Linear(per_gs_lt+lt, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 14),
        )
        
        # self.model_x = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 3),
        # )
        # self.model_a = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 1),
        # )
        # self.model_r = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 4),
        # )
        # self.model_s = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 3),
        # )
        # self.model_c = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 3),
        # )

        # self.model_x2 = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt+ltt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 3),
        # )
        # self.model_a2 = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt+ltt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 1),
        # )
        # self.model_r2 = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt+ltt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 4),
        # )
        # self.model_s2 = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt+ltt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 3),
        # )
        # self.model_c2 = nn.Sequential(
        #     nn.Linear(per_gs_lt+lt+ltt, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, d),
        #     nn.ReLU(),
        #     nn.Linear(d, 3),
        # )
        

    def forward(self, t, deg_idx):  #t, deg):
        
        #sampled_positions = F.tanh(self.pos_embedding(torch.arange(50538).to('cuda:0')))
        sampled_positions = F.tanh(self.pos_embedding(torch.arange(self.pos_embedding.num_embeddings).to('cuda:0')))
        
        time_emb = self.time_embedding(t).squeeze()
        time_view_emb = self.view_embedding[deg_idx](t).squeeze()
        

        sampled_times = F.tanh(time_emb)
        sampled_times_views = F.tanh(time_emb+time_view_emb)
        #F.tanh(self.time_embedding(t).squeeze())
        #sampled_degs = F.tanh(self.deg_embedding(deg).squeeze())

        #print(sampled_positions.shape, sampled_times.shape, sampled_degs.shape)
        x_time = torch.cat([sampled_positions,sampled_times], 1)
        x_time_view = torch.cat([sampled_positions,sampled_times_views], 1)
        #x_view = torch.cat([sampled_positions,sampled_times,sampled_degs], 1)
        d_offset_time = self.model_offset(x_time)
        d_offset_time_view = self.model_offset(x_time_view)

        d_alphas1 = d_offset_time[:,[0]]
        d_scales1 = d_offset_time[:,1:4]
        d_rots1 = d_offset_time[:,4:8]
        d_low_shs1 = d_offset_time[:,8:11]
        d_pws1 = d_offset_time[:,11:]


        d_alphas2 = d_offset_time_view[:,[0]]
        d_scales2 = d_offset_time_view[:,1:4]
        d_rots2 = d_offset_time_view[:,4:8]
        d_low_shs2 = d_offset_time_view[:,8:11]
        d_pws2 = d_offset_time_view[:,11:]

        
        return d_pws1, d_alphas1, d_scales1, d_rots1, d_low_shs1, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2
