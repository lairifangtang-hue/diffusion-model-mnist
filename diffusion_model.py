from typing import Dict, Tuple
from tqdm import tqdm
import os
os.makedirs('./data/diffusion_outputs10', exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def create_mnist_dataloaders(batch_size, image_size=32, num_workers=0):
    
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=preprocess
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root="./mnist_data", 
        train=False,
        download=True,
        transform=preprocess
    )

    return DataLoader(torch.utils.data.Subset(train_dataset, range(500)), batch_size=batch_size, shuffle=True, num_workers=num_workers),\
           DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            out = self.conv1(x)
            out = self.conv2(out)
            if self.same_channels:
                return x + out
            else:
                return out
        else:
            out = self.conv1(x)
            out = self.conv2(out)
            return out

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask))
        c = c * context_mask
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1+ temb1, down2)
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T, schedule_type='linear'):
    if schedule_type == 'linear':
        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        alphabar_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()
    elif schedule_type == 'cosine':
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = torch.cat([torch.ones(1), alphas])
        beta_t = 1 - alphas
        beta_t = torch.clamp(beta_t, 0, 0.999)
        sqrt_beta_t = torch.sqrt(beta_t)
        alphabar_t = alphas_cumprod
        alpha_t = alphas
    
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


T = 400
linear_sched = ddpm_schedules(1e-4, 0.02, T, 'linear')
cosine_sched = ddpm_schedules(1e-4, 0.02, T, 'cosine')
plt.figure(figsize=(10, 6))
plt.plot(linear_sched['alphabar_t'].numpy(), label='Linear')
plt.plot(cosine_sched['alphabar_t'].numpy(), label='Cosine')
plt.xlabel('Timestep t')
plt.ylabel('Alphabar_t')
plt.title('Comparison of Noise Schedules')
plt.legend()
plt.grid(True)
plt.savefig('./data/diffusion_outputs10/schedule_comparison.png')
plt.show()


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, schedule_type='linear'):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T, schedule_type).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)
        sqrtab = self.sqrtab[_ts].view(-1, 1, 1, 1)
        sqrtmab = self.sqrtmab[_ts].view(-1, 1, 1, 1)
        x_t = sqrtab * x + sqrtmab * noise
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):
        x_i = torch.randn(n_sample, *size).to(device)
        c_i = torch.arange(0,10).to(device)
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
        context_mask = torch.zeros_like(c_i).to(device)
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.
        x_i_store = []
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device).view(-1, 1, 1, 1).repeat(n_sample, 1, 1, 1)
            x_in = x_i.repeat(2, 1, 1, 1)
            t_in = t_is.repeat(2, 1, 1, 1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_in, c_i, t_in, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            oneover_sqrta = self.oneover_sqrta[i]
            mab_over_sqrtmab = self.mab_over_sqrtmab[i]
            sqrt_beta_t = self.sqrt_beta_t[i]
            x_i = oneover_sqrta * (x_i - mab_over_sqrtmab * eps) + sqrt_beta_t * z
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        return x_i, np.array(x_i_store)


def ddim_sample(self, n_sample, size, device, guide_w=0.0, ddim_steps=50, eta=0.0):
    device = torch.device(device)
    n_T = self.n_T
    times = torch.linspace(1, n_T, ddim_steps).long().to(device)
    times_prev = torch.cat([torch.tensor([0]).to(device), times[:-1]])
    x_i = torch.randn(n_sample, *size).to(device)
    c_i = torch.arange(0, 10).to(device)
    c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
    x_i_store = []
    for i in range(ddim_steps - 1, -1, -1):
        t_idx = times[i]
        t_prev_idx = times_prev[i]
        t_is = (torch.ones(n_sample) * t_idx / n_T).to(device).view(-1, 1, 1, 1)
        x_in = x_i.repeat(2, 1, 1, 1)
        t_in = t_is.repeat(2, 1, 1, 1)
        c_in = c_i.repeat(2)
        m_in = torch.zeros_like(c_in).to(device)
        m_in[n_sample:] = 1.
        eps_all = self.nn_model(x_in, c_in, t_in, m_in)
        eps1 = eps_all[:n_sample]
        eps2 = eps_all[n_sample:]
        eps = (1 + guide_w) * eps1 - guide_w * eps2
        ab_t = self.alphabar_t[t_idx]
        ab_prev = self.alphabar_t[t_prev_idx]
        pred_x0 = (x_i - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
        if eta > 0:
            sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
        else:
            sigma = 0
        dir_xt = torch.sqrt(1 - ab_prev - sigma**2) * eps
        noise = torch.randn_like(x_i) if i > 0 else 0
        x_i = torch.sqrt(ab_prev) * pred_x0 + dir_xt + sigma * noise
        if i % 10 == 0 or i == ddim_steps - 1:
            x_i_store.append(x_i.detach().cpu().numpy())
    return x_i, np.array(x_i_store)
DDPM.ddim_sample = ddim_sample


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_mnist(schedule_type='linear'):

    imagesize = 32
    # hardcoding these here
    n_epoch = 1
    batch_size = 256
    n_T = 50 # 500
    device = torch.device('cpu')
    n_classes = 10
    n_feat = 32 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    save_dir = './data/diffusion_outputs10/'
    ws_test = [2.0] # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1, schedule_type=schedule_type)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    dataloader, _ = create_mnist_dataloaders(batch_size=batch_size, image_size=imagesize, num_workers=0)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 1*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 32, 32), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"{schedule_type}_image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"{schedule_type}_image_ep{ep}_w{w}.png")

                if ep%5==0 or ep == int(n_epoch-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

def generate_samples(model_path, save_dir, n_samples=40, image_size=(1, 32, 32), device="cuda"):
    device = torch.device(device)
    n_classes = 10
    n_feat = 32
    n_T = 50
    model = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), 
                 betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    stuid_digits = [0, 1, 2, 3, 4, 5, 6]
    n_sample = len(stuid_digits)
    with torch.no_grad():
        x_i = torch.randn(n_sample, *image_size).to(device)
        c_i = torch.tensor(stuid_digits).to(device)
        for i in range(model.n_T, 0, -1):
            t_is = (torch.ones(n_sample) * i / model.n_T).to(device).view(-1, 1, 1, 1)
            z = torch.randn(n_sample, *image_size).to(device) if i > 1 else 0
            guide_w = 2.0
            x_in = x_i.repeat(2, 1, 1, 1)
            t_in = t_is.repeat(2, 1, 1, 1)
            c_in = c_i.repeat(2)
            m_in = torch.zeros_like(c_in).to(device)
            m_in[n_sample:] = 1.
            eps_all = model.nn_model(x_in, c_in, t_in, m_in)
            eps1 = eps_all[:n_sample]
            eps2 = eps_all[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            oneover_sqrta = model.oneover_sqrta[i]
            mab_over_sqrtmab = model.mab_over_sqrtmab[i]
            sqrt_beta_t = model.sqrt_beta_t[i]
            x_i = oneover_sqrta * (x_i - mab_over_sqrtmab * eps) + sqrt_beta_t * z
        grid = make_grid(x_i*-1 + 1, nrow=len(stuid_digits))
        save_image(grid, save_dir + "A0123456J_Manus_Assignment_4.jpg")
        print(f"Generated StuID image at {save_dir}A0123456J_Manus_Assignment_4.jpg")


if __name__ == "__main__":
    set_seed()
    print("Starting Linear Schedule Training...")
    train_mnist(schedule_type='linear')
    print("Starting Cosine Schedule Training...")
    train_mnist(schedule_type='cosine')
    # Run comparison experiments
    print("Running DDIM comparison experiments...")
    # Load the trained model for comparison
    model_path = './data/diffusion_outputs10/model_4.pth'
    device = torch.device('cpu')
    n_classes = 10
    n_feat = 32
    n_T = 50

    ddpm_model = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes),
                      betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1, schedule_type='cosine')
    ddpm_model.load_state_dict(torch.load(model_path, map_location=device))
    ddpm_model.to(device)
    ddpm_model.eval()

    # DDPM sampling (full steps)
    with torch.no_grad():
        n_sample = n_classes  # Generate one of each digit
        x_gen_ddpm, _ = ddpm_model.sample(n_sample, (1, 32, 32), device, guide_w=2.0)
        grid_ddpm = make_grid(x_gen_ddpm * -1 + 1, nrow=n_classes)
        save_image(grid_ddpm, './data/diffusion_outputs10/ddpm_full_sample.png')
        print("Saved DDPM full sample image.")

    # DDIM sampling with different step counts
    ddim_steps_list = [10, 20, 50]
    for ddim_steps in ddim_steps_list:
        with torch.no_grad():
            x_gen_ddim, _ = ddpm_model.ddim_sample(n_sample, (1, 32, 32), device, guide_w=2.0, ddim_steps=ddim_steps)
            grid_ddim = make_grid(x_gen_ddim * -1 + 1, nrow=n_classes)
            save_image(grid_ddim, f'./data/diffusion_outputs10/ddim_sample_{ddim_steps}_steps.png')
            print(f"Saved DDIM sample with {ddim_steps} steps.")

    # Generate the final required image (StuID)
    generate_samples('./data/diffusion_outputs10/model_4.pth', './data/diffusion_outputs10/', device='cpu')


################################
# Your code starts here
# Run comparison experiments and save results
################################



################################
# Your code ends here
################################
