print("===== import library ===== reduced logging")
import os
import sys
#sys.path.append('/data2/bright/my_gs')
sys.path.append('thirdparty/EasyGaussianSplatting')
#from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import look_at_view_transform
# print(look_at_view_transform)
import torch
import numpy as np
import torch.optim as optim
#import matplotlib.pyplot as plt
from gsplat.pytorch_ssim import gau_loss
from gsplat.gau_io import *
from gsplat.gausplat_dataset import *
from gsplat.gsmodel import *
torch.autograd.set_detect_anomaly(True)
from PIL import Image
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--path", help="the path of dataset")
args = parser.parse_args('')

from gsplat.gausplat import *
from PIL import Image
#import matplotlib.pyplot as plt
#from util.render import create_renderer
from util.model import model_gs, model_gs_deg, model_gs_deg_vary, model_gs_deg_time_view, model_gs_deg_time_view_6layer
from util.model_boss_larger import model_gs_deg_time_view_boss
from util.model_hierachy_new import model_gs_deg_time_view_hierachy_new, model_gs_deg_time_view_hierachy_new_viewdep, model_gs_deg_time_view_hierachy_new_viewdep_share_time_geo
from util.model_hierachy_new_2 import model_gs_deg_time_view_hierachy_new_triplane_h_pos_lowres_lowtime_triplane_level_incon, model_gs_deg_time_view_hierachy_new_triplane_h_pos_lowres_lowtime_triplane_level_incon_fix
#from pytorch3d.io import load_objs_as_meshes , load_obj
from datetime import datetime
import lpips
import pycolmap
import time as ttt
import torch.nn.functional as F
import argparse



from torch.func import jvp, vmap
# --- Utilities: quaternion/matrix and polar rotation ---
def quat_to_matrix(q):
    # q: (4,) or (...,4) unit quaternion (w,x,y,z)
    w,x,y,z = q.unbind(-1)
    ww,xx,yy,zz = w*w, x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1] + (3,3))
    return R

def matrix_to_quaternion(R):
    m00,m01,m02 = R[...,0,0],R[...,0,1],R[...,0,2]
    m10,m11,m12 = R[...,1,0],R[...,1,1],R[...,1,2]
    m20,m21,m22 = R[...,2,0],R[...,2,1],R[...,2,2]
    t = m00+m11+m22
    w = torch.sqrt(torch.clamp(t+1.0, min=0))*0.5
    x = torch.sqrt(torch.clamp(1+m00-m11-m22, min=0))*0.5
    y = torch.sqrt(torch.clamp(1-m00+m11-m22, min=0))*0.5
    z = torch.sqrt(torch.clamp(1-m00-m11+m22, min=0))*0.5
    x = torch.copysign(x, m21-m12)
    y = torch.copysign(y, m02-m20)
    z = torch.copysign(z, m10-m01)
    q = torch.stack([w,x,y,z], dim=-1)
    return q / (q.norm(dim=-1, keepdim=True) + 1e-12)


def make_phi_from_deform_constants(d_x,d_y,d_z):
    def phi(p):
        # y = p[..., 1]
        # y_norm = 1 - (y - y_min) / (y_max - y_min + eps)
        # shear_offset = c * y_norm * h
        out = p.clone()
        out[..., 0] = p[..., 0] + d_x #d_x
        out[..., 1] = p[..., 1] + d_y #d_y
        out[..., 2] = p[..., 2] + d_z #d_z
        return out
    return phi

def jvp_batched(phi_single, X, V):
    """
    phi_single: (3,) -> (3,)  (no batching inside)
    X: (N,3), V: (N,3)
    returns J·V: (N,3)

    """
    # vectorize single-sample jvp across batch
    return vmap(lambda x, v: jvp(phi_single, (x,), (v,))[1])(X, V)


def apply_shear_to_splats(d_x,d_y,d_z, pws, rots, scales , eps=1e-8,c=0.003):
    """
    pws:   (N,3) positions
    rots:  (N,4) quaternions (w,x,y,z) or (N,3,3) rotation matrices
    scales:(N,3) anisotropic scales (sx,sy,sz)
    """
    
    device, dtype = pws.device, pws.dtype
    N = pws.shape[0]

    with torch.no_grad():
        phi = make_phi_from_deform_constants(d_x,d_y,d_z)
    # Deform positions
    pws_sheared = phi(pws)


    R = quat_to_matrix(rots)                  # (N,3,3)
    M = R * scales[:, None, :]            # (N,3,3)


    #print(M.shape, torch.autograd.functional.jvp(phi,pws,M[:,0]) )
    # return 0,0,0
    
    # Mprime = torch.func.jvp(phi, pws, V)
    # m1 = torch.autograd.functional.jvp(phi,pws,M[:,0])[1].unsqueeze(1)
    # m2 = torch.autograd.functional.jvp(phi,pws,M[:,1])[1].unsqueeze(1)
    # m3 = torch.autograd.functional.jvp(phi,pws,M[:,2])[1].unsqueeze(1)
    m1 = torch.autograd.functional.jvp(phi,pws,M[:,:,0])[1].unsqueeze(2)
    m2 = torch.autograd.functional.jvp(phi,pws,M[:,:,1])[1].unsqueeze(2)
    m3 = torch.autograd.functional.jvp(phi,pws,M[:,:,2])[1].unsqueeze(2)

    Mprime = torch.cat([m1,m2,m3],2)
    # Mprime = torch.cat([m1,m2,m3],1)
    # return 0, 0, 0
    
    # Mprime = vmap(lambda V: jvp_batched(phi, pws, V))(M.permute(2, 0, 1)).permute(1, 2, 0)  # (N,3,3)

    U, S, Vh = torch.linalg.svd(Mprime, full_matrices=False)  # S is sorted desc.

    # Ensure U is a proper rotation (det +1); flipping a column keeps A A^T the same.
    detU = torch.det(U)
    need_flip = (detU < 0.0)[..., None, None]          # (N,1,1)
    if torch.any(need_flip):
        # flip the last column of U wherever needed
        flip = U.new_ones(U.shape[:-2] + (3,))
        flip[..., -1] = -1.0
        U = torch.where(need_flip, U * flip.unsqueeze(-2), U)

    R_new = U                                          # (N,3,3)
    s_new = S                                          # (N,3) singular values ≥ 0

    rots_out = matrix_to_quaternion(R_new)         # returns unit (w,x,y,z)
    scales_out = s_new

    return pws_sheared, rots_out, scales_out


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--save_folder", type=str, default="V100", help="Path for weights and logs")
parser.add_argument("--participant", type=str, default="ch", help="Participant name")
parser.add_argument("--sen", type=str, default="obama_real_tttrim", help="Sentence name")
parser.add_argument("--lt", type=int, default=256, help="LT")
parser.add_argument("--ltt", type=int, default=64, help="LTT")
parser.add_argument("--per_gs_lt", type=int, default=64, help="Per GS LT")
parser.add_argument("--d", type=int, default=256, help="D")
parser.add_argument("--lr_multiplier", type=float, default=0.001, help="Learning rate multiplier")
parser.add_argument("--epoch", type=int, default=60000, help="Epoch")
parser.add_argument("--gres", type=int, default=4, help="gres")
parser.add_argument("--lr", type=float, default=0.0001, help="lr")
# parser.add_argument("--ratio", type=float, default=1.0/15.0, help="lr")
# parser.add_argument("--num_v", type=int, default=100, help="Number of views")

args = parser.parse_args()

model_path = args.save_folder
participant = args.participant
sen = args.sen
lt = args.lt
ltt = args.ltt
per_gs_lt = args.per_gs_lt
d = args.d
lr_multiplier = args.lr_multiplier
epoch = args.epoch
gres = args.gres
lr = args.lr
epoch = args.epoch
# ratio = args.ratio
# participant = 'nersemble_304'
# sen = 'exp1_multi_restruct'
# #'A_synctalk_heygen_hierachy'
# lt = 512
# ltt = args.ltt
# per_gs_lt = args.per_gs_lt
# d = args.d
# lr_multiplier = args.lr_multiplier
# epoch = args.epoch


print("===== prep data =====")
device = 'cuda:0'
loss_fn_alex = lpips.LPIPS(net='vgg').to(device) #lpips.LPIPS(net='vgg',spatial='on').to(device)
#lpips.LPIPS(net='alex',spatial='on').to(device)
# gs = load_gs(f'gs_data/{participant}_final2/final.npy')
finetune_base_model_path = f'my_new_img/A_iv253_debug_restrict_identity_gs_identity_change_model_static_color_extend_openteeth_oct24/{participant}/weight'
gs = load_gs(f'{finetune_base_model_path}/6.npy')

degs = np.linspace(-75, 75, 101)
deg_idxs = list(range(101))
frame_root = f"prep_data/{sen}/{participant}/img"
#f"prep_data/A_synctalk/{participant}_fullreso_amp12_fixr/img"
# frame_root = "/data2/bright/LivePortrait/my_gs_data/pbert_rotate_100views_all/img"
#"/data2/bright/LivePortrait/my_gs_data/pbert_d3_21_all_img/img"
#"/data2/bright/LivePortrait/pbert_rotate_90_bg_21_exp_img"
# views = sorted(os.listdir(frame_root))
#time_idx = (4-len(str(time)))*'0'+str(time)
learning_rate = 1e-4
training_params, adam_params = get_training_params(gs)
training_params['pws'].requires_grad = True
training_params['alphas_raw'].requires_grad = True
training_params['scales_raw'].requires_grad = True
training_params['rots_raw'].requires_grad = True
training_params['low_shs'].requires_grad = True
training_params['high_shs'].requires_grad = False

xyz_coord = training_params['pws'].cpu().detach().numpy()
xyz_coord = (xyz_coord-xyz_coord.min(0))/(xyz_coord.max(0)-xyz_coord.min(0))*2-1
xyz_coord = torch.Tensor(xyz_coord).unsqueeze(0).unsqueeze(0).to(device)

cams = os.listdir(f"prep_data/{sen}/{participant}/img")
t= len(os.listdir(f"prep_data/{sen}/{participant}/img/{cams[0]}")) #600
# t_model=1000
print(participant, sen,t,training_params['pws'].shape[0])
model = model_gs_deg_time_view_hierachy_new_triplane_h_pos_lowres_lowtime_triplane_level_incon_fix(lt=lt,ltt=ltt,per_gs_lt=per_gs_lt,d=d,t=t,deg_s=120,num_v=training_params['pws'].shape[0],gres=gres).to(device)

adam_params = adam_params[:2]+adam_params[3:]
for adam_param in adam_params:
    adam_param['lr'] = adam_param['lr']*lr_multiplier

adam_params = adam_params[:2]+adam_params[3:]
model_dict = {}
model_dict['params'] = list(model.parameters())
model_dict['lr'] = 1e-4
model_dict['name'] = "mlp"
adam_params.append(model_dict)
# adam_params = [adam_params[-1]]
optimizer = torch.optim.Adam(adam_params, lr=0.0)
num_v = training_params['pws'].shape[0]
frames = sorted(os.listdir(os.path.join(frame_root, cams[0])))
reconstruction = pycolmap.Reconstruction(f"prep_data/liveportrait/{participant}/colmap_gs/sparse/0")
live_root=f"prep_data/{sen}/{participant}/img"
data = [(image,camera) for (image_id, image), (camera_id, camera) in zip(reconstruction.images.items(), reconstruction.cameras.items())]
fx,fy,cx,cy = data[0][1].params
w,h = data[0][1].width, data[0][1].height
# data = [image for (image_id, image) in reconstruction.images.items() if '037' not in image.name ]
# = [image for (image_id, image) in reconstruction.images.items() if '191' not in image.name ]
data = [image for (image_id, image) in reconstruction.images.items() if '042' in image.name or '048' in image.name or '191' in image.name or '043' in image.name or '007' in image.name ]


font_idx = 0
for i in range(len(data)):
    if '191' in data[i].name:
        font_idx = i
        break

for i in range(5):
    model.upsample_latent()
model.load_state_dict(torch.load(f'{finetune_base_model_path}/6.pth', weights_only=True))
for i in range(2):
    model.upsample_latent()

def quaternion_multiply(q, r):
    """
    Batch quaternion multiplication (Hamilton product): q * r

    Parameters:
        q: Tensor of shape (N, 4) — [x, y, z, w]
        r: Tensor of shape (N, 4) — [x, y, z, w]
    
    Returns:
        Tensor of shape (N, 4): q * r
    """
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack((w, x, y, z), dim=-1)

q_initial = torch.Tensor([1.0,0.0,0.0,0.0]).unsqueeze(0).to(device)

save_p = 5000
log_p  = 100
cano_op_less_than = 10000
print("===== start training =====")

acc_losses = []
acc_loss = []
acc_losses2 = []
acc_loss2 = []
t1_acc = []
t2_acc = []
t3_acc = []
t4_acc = []
dist = 6
xyz_mean = training_params['pws'].mean(0).cpu().detach().numpy()
hhlevels = [32,16,8,4,2,1]
cc = 0
#print(pytorch3d.__dir__(),pytorch3d.__version__)
# model_path = f'c2f_{participant}_{sen}'
#'c2f_viewdep_experiment_identity_changmodel2_staticcolor_multivideo_gres4_2'
#'c2f_viewdep_experiment_identity_changmodel2_staticcolor_swaprot'

if not os.path.exists(f'my_new_img/{model_path}'):
    os.makedirs(f'my_new_img/{model_path}')
    
if not os.path.exists(f'my_new_img/{model_path}/img/'):
    os.makedirs(f'my_new_img/{model_path}/img/')
    
if not os.path.exists(f'my_new_img/{model_path}/img/validate'):
    os.makedirs(f'my_new_img/{model_path}/img/validate')
    
if not os.path.exists(f'my_new_img/{model_path}/weight'):
    os.makedirs(f'my_new_img/{model_path}/weight')

    
for hh_level in [6]:
    optimizer = optim.AdamW([
        {"params": model.grid_xy},         # special layer
        {"params": model.grid_xz},
        {"params": model.grid_yz},
        {"params": model.time_module.parameters()},
        {"params": model.time_modules.parameters()},
        {"params": [training_params['alphas_raw']]},
        {"params": [training_params['low_shs']]},      
        {"params": model.model_x.parameters()},      
        {"params": model.model_a.parameters()},  
        {"params": model.model_s.parameters()},  
        {"params": model.model_r.parameters()},  # default
        {"params": model.model_c.parameters()},
    ], lr=lr, weight_decay=1e-2)
    
    acc_loss = []
    losses = []
    for i in range(epoch):
        st1 =  ttt.time()
        if (i+1)%10==0:
            deg_idx = font_idx
        else:
            # rand_val = np.random.rand()
            # if rand_val<=ratio:
            #     deg_idx=14
            # else:
            deg_idx = np.random.choice(range(len(data))) #2 #np.random.choice(range(len(data)))
            #deg_idx = font_idx #np.random.choice(range(len(data)))
        deg = degs[deg_idx]
        # view = os.path.join(frame_root, views[deg_idx])
        deg_str = (4-len(str(deg_idx)))*'0'+str(deg_idx)
        deg_tensor = deg_idx #torch.Tensor(np.array([deg_idx])).long().unsqueeze(0).expand(num_v,1).unsqueeze(0).to(device)
        if (i+1)%10==0:
            time = 61 #600+1
        else:
            time = np.random.choice(t)+1 #np.random.choice(list(range(550,615))) + 1 #np.random.choice(t)+1 #90 #600 #90  #154 #i+1 #np.random.choice(t)+1
        time_idx = (4-len(str(time)))*'0'+str(time)
        time_tensor = (time-1)/t*2-1
        #torch.Tensor(np.array([time-1])).long().unsqueeze(0).expand(num_v,1).un squeeze(0).to(device)

        if (i+1)%10==0:
            image = data[deg_idx]
            #data2[0]
        else:
            image = data[deg_idx]
        #image = data[2] #data[deg_idx] #data[2] #data[deg_idx]
        img_folder = image.name.split('.')[0]
        img_folder = img_folder.split('_')[-2]
        img_folder = img_folder[-3:]
        img_folder = participant.split('_')[-1]+"_"+img_folder
        
        frame_path = os.path.join(live_root, img_folder, time_idx+'.png')
        pt3d_img = Image.open(frame_path)
        pt3d_img = torch.Tensor(np.array(pt3d_img)/255).to(device)
        pt3d_img = pt3d_img.permute([2,0,1])
    
    
        R = torch.Tensor(image.cam_from_world.rotation.matrix())
        T = torch.Tensor(image.cam_from_world.translation)
        cam = Camera(0,w,h,fx, fy, cx, cy, torch.Tensor(R).to(device), torch.Tensor(T).to(device),'')
        en1 =  ttt.time()
        st2 = ttt.time()
        d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2 = model(xyz_coord, time_tensor, deg_tensor, hh_level) #, deg_tensor) #model(time_tensor, deg_tensor)
        canonical_pws = training_params['pws'].detach()
        canonical_rots = get_rots(training_params['rots_raw'].detach())
        canonical_scales = get_scales(training_params['scales_raw'].detach())
        pws2, rots2, scales2 = apply_shear_to_splats(d_pws[:,0],d_pws[:,1],d_pws[:,2],canonical_pws, canonical_rots, canonical_scales , eps=1e-8,c=0.003)
        scales2 = get_scales(get_scales_raw(scales2)+d_scales)
        d_rots = get_rots(q_initial+d_rots)
        #d_rots = get_rots(d_rots)
        # rots2 = quaternion_multiply(rots2, d_rots)
        rots2 = quaternion_multiply(d_rots, rots2)
        
        en2 = ttt.time()
    
        st3 = ttt.time()
        
        #pws2 = training_params['pws'].detach()+d_pws
        us = torch.zeros([pws2.shape[0], 2], dtype=torch.float32, device='cuda', requires_grad=True)
        alphas2 = get_alphas(training_params['alphas_raw'])
        shs2 = get_shs(training_params['low_shs'], training_params['high_shs'])
    
        gs_image_time, mask = GSFunction.apply(pws2, shs2, alphas2, scales2, rots2, us, cam)
        gs_image_time_down = torch.nn.functional.interpolate(gs_image_time.unsqueeze(0),scale_factor=1/hhlevels[hh_level-1], mode='area').squeeze()
        pt3d_img_down = torch.nn.functional.interpolate(pt3d_img.unsqueeze(0),scale_factor=1/hhlevels[hh_level-1], mode='area').squeeze()
    
        gs_image_time_up = F.interpolate(gs_image_time_down.unsqueeze(0), size=(802, 550), mode="bilinear", align_corners=False).squeeze()
        pt3d_img_up = F.interpolate(pt3d_img_down.unsqueeze(0), size=(802, 550), mode="bilinear", align_corners=False).squeeze()
        gs_loss = gau_loss(gs_image_time_down, pt3d_img_down)

        #loss2 = gs_loss_lv1+ gs_loss_lv2 + gs_loss_lv3 + gs_loss_lv4 + gs_loss_lv5 + F.relu(d_scales - 2.0).norm(dim=1).mean()*0.1
        loss2 = gs_loss # + F.relu(d_scales - 2.0).norm(dim=1).mean()*0.2
        loss = loss2 # loss1+loss2

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad()
        acc_loss.append(loss.item())
        if (i+1)%10==0:
            losses.append(np.mean(acc_loss))
            acc_loss = []
            cc_str = f"{cc:04d}"
            cc+=1
            # ppath = f'visualize_gs/4_pergaussian_latent/{cc_str}.ply'
            # save_ply(ppath, dtype_full)
            img1 = pt3d_img_up.permute([1,2,0]).cpu().detach().numpy()
            img2 = gs_image_time_up.permute([1,2,0]).cpu().detach().numpy()
            img3 = gs_image_time.permute([1,2,0]).cpu().detach().numpy()
            imgsss = np.clip(np.concatenate([img1,img2,img3],1),0,1)
            imgsss = Image.fromarray(np.uint8(imgsss*255))
            imgsss.save(f'my_new_img/{model_path}/img/validate/{cc:04d}.png')
            # plt.imshow(imgsss)
            # plt.show()
        if (i+1)%200==0:
            print(hh_level, i, losses[-1])
    print("====================================")
    torch.save(model.state_dict(), f'my_new_img/{model_path}/weight/{hh_level}.pth')
    gs_path = f'my_new_img/{model_path}/weight/{hh_level}.npy'
    save_training_params(gs_path, training_params)
    if not os.path.exists(f'my_new_img/{model_path}/img/{hh_level}'):
        os.mkdir(f'my_new_img/{model_path}/img/{hh_level}')
    for i in range(t):
        deg_idx = font_idx
        deg = degs[deg_idx]
        # view = os.path.join(frame_root, views[deg_idx])
        deg_str = (4-len(str(deg_idx)))*'0'+str(deg_idx)
        deg_tensor = deg_idx #torch.Tensor(np.array([deg_idx])).long().unsqueeze(0).expand(num_v,1).unsqueeze(0).to(device)
        time = i+1 #np.random.choice(t)+1 #90 #600 #90  #154 #i+1 #np.random.choice(t)+1
        time_idx = (4-len(str(time)))*'0'+str(time)
        time_tensor = (time-1)/t*2-1
        #torch.Tensor(np.array([time-1])).long().unsqueeze(0).expand(num_v,1).un squeeze(0).to(device)
    
        image = data[deg_idx] #data[deg_idx] #data[2] #data[deg_idx]
        img_folder = image.name.split('.')[0]
        img_folder = img_folder.split('_')[-2]
        img_folder = img_folder[-3:]
        img_folder = participant.split('_')[-1]+"_"+img_folder
        
        frame_path = os.path.join(live_root, img_folder, time_idx+'.png')
        pt3d_img = Image.open(frame_path)
        pt3d_img = torch.Tensor(np.array(pt3d_img)/255).to(device)
        pt3d_img = pt3d_img.permute([2,0,1])
    
    
        R = torch.Tensor(image.cam_from_world.rotation.matrix())
        T = torch.Tensor(image.cam_from_world.translation)
        cam = Camera(0,w,h,fx, fy, cx, cy, torch.Tensor(R).to(device), torch.Tensor(T).to(device),'')
        en1 =  ttt.time()
        st2 = ttt.time()
        d_pws, d_alphas, d_scales, d_rots, d_low_shs, d_pws2, d_alphas2, d_scales2, d_rots2, d_low_shs2 = model(xyz_coord, time_tensor, deg_tensor, hh_level) #, deg_tensor) #model(time_tensor, deg_tensor)
        canonical_pws = training_params['pws'].detach()
        canonical_rots = get_rots(training_params['rots_raw'].detach())
        canonical_scales = get_scales(training_params['scales_raw'].detach())
        pws2, rots2, scales2 = apply_shear_to_splats(d_pws[:,0],d_pws[:,1],d_pws[:,2],canonical_pws, canonical_rots, canonical_scales , eps=1e-8,c=0.003)
        scales2 = get_scales(get_scales_raw(scales2)+d_scales)
        d_rots = get_rots(q_initial+d_rots)
        rots2 = quaternion_multiply(d_rots, rots2)
        en2 = ttt.time()
    
        st3 = ttt.time()
        us = torch.zeros([pws2.shape[0], 2], dtype=torch.float32, device='cuda', requires_grad=True)
        alphas2 = get_alphas(training_params['alphas_raw'])
        shs2 = get_shs(training_params['low_shs'], training_params['high_shs'])
    
        gs_image_time, mask = GSFunction.apply(pws2, shs2, alphas2, scales2, rots2, us, cam)
        gs_image_time_down = torch.nn.functional.interpolate(gs_image_time.unsqueeze(0),scale_factor=1/hhlevels[hh_level-1], mode='area').squeeze()
        pt3d_img_down = torch.nn.functional.interpolate(pt3d_img.unsqueeze(0),scale_factor=1/hhlevels[hh_level-1], mode='area').squeeze()
    
        gs_image_time_up = F.interpolate(gs_image_time_down.unsqueeze(0), size=(802, 550), mode="bilinear", align_corners=False).squeeze()
        pt3d_img_up = F.interpolate(pt3d_img_down.unsqueeze(0), size=(802, 550), mode="bilinear", align_corners=False).squeeze()
        img1 = pt3d_img_up.permute([1,2,0]).cpu().detach().numpy()
        img2 = gs_image_time_up.permute([1,2,0]).cpu().detach().numpy()
        img3 = gs_image_time.permute([1,2,0]).cpu().detach().numpy()
        imgsss = np.clip(np.concatenate([img1,img2,img3],1),0,1)
        imgsss = Image.fromarray(np.uint8(imgsss*255))
        imgsss.save(f'my_new_img/{model_path}/img/{hh_level}/{i:04d}.png')
    if hh_level!=6:
        model.upsample_latent()