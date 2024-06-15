import torch
import gdown
import zipfile


def download_and_extract(url, output_filepath, extract_dir):
    gdown.download(url, output_filepath)
    print(f'Download data from {url} to {output_filepath}')
    with zipfile.ZipFile(output_filepath) as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f'Successfully extracted {output_filepath} to {extract_dir}')


def save_checkpoint(model1, optimizer, file_name, model2=None):
    checkpoint = {
        'model1': model1.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if model2:
        checkpoint['model2'] = model2.state_dict()

    torch.save(checkpoint, file_name)
    print(f"Checkpoint saved to {file_name}")


def load_checkpoint(model1, optimizer, file_name, device, lr=0.0002, model2=None):
    checkpoint = torch.load(file_name, map_location=device)
    model1.load_state_dict(checkpoint['model1'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if model2:
        model2.load_state_dict(checkpoint['model2'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"Checkpoint loaded from {file_name}")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_disc_loss(real, fake, disc, adv_criterion):
    d_fake = disc(fake.detach())
    d_fake_loss = adv_criterion(d_fake, torch.zeros_like(d_fake))
    d_real = disc(real)
    d_real_loss = adv_criterion(d_real, torch.ones_like(d_real))

    disc_loss = (d_fake_loss + d_real_loss) / 2
    return disc_loss


def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    fake_Y = gen_XY(real_X)
    disc_fake = disc_Y(fake_Y)
    adv_loss = adv_criterion(disc_fake, torch.ones_like(disc_fake))

    return adv_loss, fake_Y


def get_identity_loss(real_X, gen_YX, identity_criterion):
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(identity_X, real_X)

    return identity_loss


def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X)

    return cycle_loss, cycle_X


def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion,
                 identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
    adv_loss_BA, fake_A = get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    adv_loss_AB, fake_B = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    gen_adv_loss = (adv_loss_AB + adv_loss_BA) / 2

    identity_loss_A = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_B = get_identity_loss(real_B, gen_AB, identity_criterion)
    gen_identity_loss = (identity_loss_A + identity_loss_B) / 2

    cycle_loss_BA, cycle_A = get_cycle_consistency_loss(
        real_A, fake_B, gen_BA, cycle_criterion
    )
    cycle_loss_AB, cycle_B = get_cycle_consistency_loss(
        real_B, fake_A, gen_AB, cycle_criterion
    )
    gen_cycle_loss = (cycle_loss_BA + cycle_loss_AB) / 2

    gen_loss = lambda_identity * gen_identity_loss + lambda_cycle * gen_cycle_loss + gen_adv_loss
    return gen_loss, fake_A, fake_B
