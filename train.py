import torch
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from IPython.display import display
from tqdm.auto import tqdm
from dataset.dataset import ImageDataset
from models.generator import Generator
from models.discriminator import Discriminator
from config import Config
from utils import *
import os


def main():
    CONFIG = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prefix = 'https://drive.google.com/uc?/export=download&id='
    file_id = CONFIG.SOURCE_URL.split('/')[-2]
    os.makedirs(CONFIG.ROOT_DIR, exist_ok=True)
    download_and_extract(prefix + file_id, CONFIG.OUTPUT_PATH, CONFIG.ROOT_DIR)

    transforms_ = transforms.Compose([
        transforms.Resize(int(CONFIG.IMG_SIZE * 1.12), Image.BICUBIC),
        transforms.RandomCrop((CONFIG.IMG_SIZE, CONFIG.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = ImageDataset(
        CONFIG.DIM_A,
        CONFIG.DIR_B,
        transforms_
    )
    loader = DataLoader(
        dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=True
    )
    print(f'Length of loader {len(loader)}')

    gen_AB = Generator(CONFIG.DIM_A, CONFIG.DIM_B).to(device)
    gen_BA = Generator(CONFIG.DIM_A, CONFIG.DIM_B).to(device)
    gen_opt = torch.optim.Adam(
        list(gen_AB.parameters()) + list(gen_BA.parameters()),
        lr=CONFIG.LR, betas=(CONFIG.BETA1, CONFIG.BETA2)
    )

    disc_A = Discriminator(CONFIG.DIM_A).to(device)
    disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=CONFIG.LR, betas=(CONFIG.BETA1, CONFIG.BETA2))
    disc_B = Discriminator(CONFIG.DIM_A).to(device)
    disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=CONFIG.LR, betas=(CONFIG.BETA1, CONFIG.BETA2))

    if CONFIG.PRETRAINED:
        load_checkpoint(gen_AB, gen_opt, CONFIG.CHECKPOINT_GEN_PATH, device, CONFIG.LR, gen_BA)
        load_checkpoint(disc_A, disc_A_opt, CONFIG.CHECKPOINT_DISC_A_PATH, device, CONFIG.LR)
        load_checkpoint(disc_B, disc_B_opt, CONFIG.CHECKPOINT_DISC_B_PATH, device, CONFIG.LR)
    else:
        gen_AB.apply(weights_init)
        gen_BA.apply(weights_init)
        disc_A.apply(weights_init)
        disc_B.apply(weights_init)
        print('Applied weights initialization')

    g_scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    d_A_scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    d_B_scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    gen_scheduler = StepLR(gen_opt, step_size=5, gamma=0.5)
    disc_A_scheduler = StepLR(disc_A_opt, step_size=5, gamma=0.5)
    disc_B_scheduler = StepLR(disc_B_opt, step_size=5, gamma=0.5)

    adv_criterion = nn.MSELoss()
    recon_criterion = nn.L1Loss()

    writer = SummaryWriter(f'logs')

    for epoch in tqdm(range(CONFIG.N_EPOCHS)):
        loop = tqdm(loader, desc=f'Epoch {epoch + 1}')
        for batch_idx, (real_A, real_B) in enumerate(loop):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Train Generators: Gab and Gba
            with torch.cuda.amp.autocast():
                gen_loss, fake_A, fake_B = get_gen_loss(
                    real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion,
                    recon_criterion, recon_criterion, CONFIG.LAMBDA_IDENTITY, CONFIG.LAMBDA_CYCLE
                )

            gen_opt.zero_grad()
            if g_scaler:
                g_scaler.scale(gen_loss).backward()
                g_scaler.step(gen_opt)
                g_scaler.update()
            else:
                gen_loss.backward()
                gen_opt.step()

            # Train Discriminators A
            with torch.cuda.amp.autocast():
                disc_A_loss = get_disc_loss(real_A, fake_A, disc_A, adv_criterion)

            disc_A_opt.zero_grad()
            if d_A_scaler:
                d_A_scaler.scale(disc_A_loss).backward()
                d_A_scaler.step(disc_A_opt)
                d_A_scaler.update()
            else:
                disc_A_loss.backward()
                disc_A_opt.step()

            # Train Discriminators B
            with torch.cuda.amp.autocast():
                disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)

            disc_B_opt.zero_grad()
            if d_B_scaler:
                d_B_scaler.scale(disc_B_loss).backward()
                d_B_scaler.step(disc_B_opt)
                d_B_scaler.update()
            else:
                disc_B_loss.backward()
                disc_B_opt.step()

            disc_A_loss = disc_A_loss.item()
            disc_B_loss = disc_B_loss.item()
            gen_loss = gen_loss.item()

            # Logging the losses
            writer.add_scalar('Discriminator/A_loss', disc_A_loss, epoch * len(loader) + batch_idx)
            writer.add_scalar('Discriminator/B_loss', disc_B_loss, epoch * len(loader) + batch_idx)
            writer.add_scalar('Generator/Total_loss', gen_loss, epoch * len(loader) + batch_idx)

            # Update progress bar
            loop.set_postfix(
                D_A_loss=disc_A_loss,
                D_B_loss=disc_B_loss,
                G_loss=gen_loss,
                lr=gen_opt.param_groups[0]['lr']
            )

            # Display generated images periodically
            if batch_idx > 0 and batch_idx % ((len(loader) - 1) // 3) == 0:
                with torch.no_grad():
                    fake_A = gen_BA(real_B)
                    fake_B = gen_AB(real_A)
                imgs = make_grid([real_A[0], fake_B[0], real_B[0], fake_A[0]], nrow=4, normalize=True)
                display(transforms.ToPILImage()(imgs))

        # Save models
        save_checkpoint(gen_AB, gen_opt, 'cycleGAN-gen-monet.pth', gen_BA)
        save_checkpoint(disc_A, disc_A_opt, "cycleGAN-discA-monet.pth")
        save_checkpoint(disc_B, disc_B_opt, "cycleGAN-discB-monet.pth")

        gen_scheduler.step()
        disc_A_scheduler.step()
        disc_B_scheduler.step()
