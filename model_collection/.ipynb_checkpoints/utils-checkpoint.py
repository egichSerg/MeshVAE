import torch, os
import numpy as np

from torch.utils.checkpoint import checkpoint_sequential

from tqdm import tqdm
from pathlib import Path

from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras)
from torchvision.transforms import Resize

from model_collection.VAE_loss.VaeLoss import VAELoss


def huber_loss(x, y, scaling=0.1):
    diff_sq = (x-y) ** 2
    loss = ((1 + diff_sq / (scaling ** 2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss
    

def max_VRAM_optim_train_step(model, target_cameras, render_bs, optimizer, criterion, dataloader, device, scheduler):
    model.train()
    encoder_modules = model.get_encoder_modules()
    decoder_modules = model.get_decoder_modules()
    
    encoder_segments = len(encoder_modules)
    decoder_segments = len(decoder_modules)
    avg_loss = 0.

    render_size = model.render_size
    resize = Resize((render_size, render_size))
    batch_size = render_bs
    for i, input_renders in tqdm(enumerate(dataloader), total=len(dataloader)):
        target_renders, target_silhouttes = input_renders.to(device).split([1, 1], dim = 1)
        target_renders, target_silhouttes = target_renders.squeeze().unsqueeze(2), target_silhouttes.squeeze().unsqueeze(2)
        mu, logvar, output_encoded = checkpoint_sequential(encoder_modules, encoder_segments, target_renders)
        output_encoded = output_encoded.unsqueeze(2)
        output_models = checkpoint_sequential(decoder_modules, decoder_segments, output_encoded)

        #make random renders
        batch_idx = torch.randperm(40)[:batch_size]
        batch_cameras = FoVPerspectiveCameras(
            R = target_cameras.R[batch_idx],
            T = target_cameras.T[batch_idx],
            zfar = target_cameras.zfar[batch_idx],
            aspect_ratio = target_cameras.aspect_ratio[batch_idx],
            fov = target_cameras.fov[batch_idx],
            device = device
        )
        
        rendered_images_silhouettes = model.get_renders(output_models, batch_cameras)
        rendered_images, rendered_silhouttes = rendered_images_silhouettes.split([1, 1], dim = -1)
        rendered_images, rendered_silhouttes = rendered_images.permute(0, 1, 4, 2, 3), rendered_silhouttes.permute(0, 1, 4, 2, 3)

        del target_renders
        del target_silhouttes
        torch.cuda.empty_cache()
        
        target_renders, target_silhouttes = input_renders.to(device).squeeze()[:, :, batch_idx].split([1, 1], dim = 1)
        target_renders, target_silhouttes = resize(target_renders.squeeze()), resize(target_silhouttes.squeeze())
        target_renders, target_silhouttes = target_renders.unsqueeze(2), target_silhouttes.unsqueeze(2)

        # print('mu', mu.shape)
        # print('logvar', logvar.shape)
        # print('rendered_images', rendered_images.shape)
        # print('rendered_silhouttes', rendered_silhouttes.shape)
        # print('target_renders', target_renders.shape)
        # print('target_silhouttes', target_silhouttes.shape)
        loss = criterion(mu, logvar, rendered_images, rendered_silhouttes, target_renders, target_silhouttes, dataloader.batch_size)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not scheduler is None:
            scheduler.step()

        avg_loss += loss.item()

    avg_loss = avg_loss / len(dataloader)
    return avg_loss

def max_VRAM_optim_eval_step(model, target_cameras, render_bs, criterion, dataloader, device):
    model.eval()
    avg_loss = 0.

    render_size = model.render_size
    resize = Resize((render_size, render_size))
    batch_size = render_bs

    with torch.inference_mode():
        for input_renders in tqdm(dataloader):
            target_renders, target_silhouttes = input_renders.to(device).split([1, 1], dim = 1)
            target_renders, target_silhouttes = target_renders.squeeze().unsqueeze(2), target_silhouttes.squeeze().unsqueeze(2)
            mu, logvar, output_models = model(target_renders)
    
            #make random renders
            batch_idx = torch.randperm(40)[:batch_size]
            batch_cameras = FoVPerspectiveCameras(
                R = target_cameras.R[batch_idx],
                T = target_cameras.T[batch_idx],
                zfar = target_cameras.zfar[batch_idx],
                aspect_ratio = target_cameras.aspect_ratio[batch_idx],
                fov = target_cameras.fov[batch_idx],
                device = device
            )
            
            rendered_images_silhouettes = model.get_renders(output_models, batch_cameras)
            rendered_images, rendered_silhouttes = rendered_images_silhouettes.split([1, 1], dim = -1)
            rendered_images, rendered_silhouttes = rendered_images.permute(0, 1, 4, 2, 3), rendered_silhouttes.permute(0, 1, 4, 2, 3)
    
            del target_renders
            del target_silhouttes
            torch.cuda.empty_cache()
        
            target_renders, target_silhouttes = input_renders.to(device).squeeze()[:, :, batch_idx].split([1, 1], dim = 1)
            target_renders, target_silhouttes = resize(target_renders.squeeze()), resize(target_silhouttes.squeeze())
            target_renders, target_silhouttes = target_renders.unsqueeze(2), target_silhouttes.unsqueeze(2)
            
            # sil_err = criterion(rendered_silhouttes, target_silhouttes).abs().mean()
            # color_err = criterion(rendered_images, target_renders).abs().mean()
            # loss = color_err + sil_err
            loss = criterion(mu, logvar, rendered_images, rendered_silhouttes, target_renders, target_silhouttes, dataloader.batch_size)
            
    
            avg_loss += loss.item()

        avg_loss = avg_loss / len(dataloader)
    return avg_loss


def max_VRAM_optim_fit(model, optimizer, criterion, train_dl, test_dl, device, epochs, train_launch, scheduler=None,
                       save_weights=True,weights_save_dir=Path('.'), early_stopping=True, early_stopping_tolerance = 5,
                       restore_best_weights=True, save_loss_history=True, overfit_mode=False):

    MODEL_SAVE_PATH = weights_save_dir
    MODEL_SAVE_PATH = weights_save_dir / model.__class__.__name__ / f'{train_launch:02d}'
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    min_save_loss = 1000000.
    
    if early_stopping:
        cache_weights_path = Path('.') / 'cached_weights.pt'
        tolerance = early_stopping_tolerance
        min_loss = 1000000.

    if save_loss_history:
        loss_history = []

    # init rendering
    num_views, azimuth_range = 40, 180.0
    elev = torch.linspace(0, 0, num_views)  # keep constant
    azim = torch.linspace(-azimuth_range, azimuth_range, num_views) + 180.0
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    target_cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    target_cameras = target_cameras.to(device)
    render_batching_size = 40

    # the training cycle
    model.to(device)
    for epoch in range(epochs):
        print(f'==========Epoch {epoch+1}==========')
        print('Training...')
        train_loss = max_VRAM_optim_train_step(model, target_cameras, render_batching_size, optimizer, criterion, train_dl, device, scheduler)        
        
        print('Validating...')
        test_loss = max_VRAM_optim_eval_step(model, target_cameras, render_batching_size, criterion, test_dl, device)

        print(f'Train loss: {train_loss:.02f} | Test loss: {test_loss:.02f}')

        if save_loss_history:
            loss_history.append([train_loss, test_loss])

        cur_loss = train_loss if overfit_mode else test_loss
        if save_weights:
            if min_save_loss > cur_loss:
                min_save_loss = cur_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH / 'best.pt')
                print(f'Saved weights to {MODEL_SAVE_PATH / "best.pt"}')
            else:
                torch.save(model.state_dict(), MODEL_SAVE_PATH / 'last.pt')
                print(f'Saved weights to {MODEL_SAVE_PATH / "last.pt"}')

        if early_stopping:
            if min_loss > cur_loss:
                min_loss = cur_loss
                tolerance = early_stopping_tolerance
                torch.save(model.state_dict(), cache_weights_path)
            else:
                tolerance -= 1
                if tolerance <= 0:
                    if restore_best_weights:
                        model.load_state_dict(torch.load(f=cache_weights_path))
                        model.eval()
                        os.remove(cache_weights_path)
                        print('Restored best weights')
                    if save_loss_history:
                        np.save(MODEL_SAVE_PATH / 'loss_history.pt', np.array(loss_history))
                    return

    if save_loss_history:
        np.save(MODEL_SAVE_PATH / 'loss_history.pt', np.array(loss_history))