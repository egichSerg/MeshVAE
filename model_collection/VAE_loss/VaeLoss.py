import torch
import torch.nn as nn

from model_collection.VAE_loss.vgg19 import VGG19
from pathlib import Path

class VAELoss(nn.Module):
    def __init__(self, device, clamp, clamp_threshold, loss_fn, KLD_coef=1., reconstruction_coef=1., perceptional_loss=True, p_loss_coef=1.):
        super(VAELoss, self).__init__()
        self.criterion = loss_fn
        
        self.KLD_coef = KLD_coef
        self.reconstruction_coef = reconstruction_coef
        self.p_loss_coef = p_loss_coef

        self.perceptional_loss = perceptional_loss

        self.clamp = clamp
        self.clamp_threshold = clamp_threshold

        wpath = Path('./models') / 'VGG19'
        self.feature_extractor_col = VGG19(weights_path=wpath / '22' / 'best.pt').to(device)
        self.feature_extractor_sil = VGG19(weights_path=wpath / '21' / 'best.pt').to(device)

    # question: how is the loss function using the mu and variance?
    def forward(self, mu, logvar, rendered_images, rendered_silhouttes, target_renders, target_silhouttes, batch_size):
        """gives the batch normalized Variational Error."""

        col_err = self.criterion(target_renders, rendered_images).abs().mean()
        sil_err = self.criterion(target_silhouttes, rendered_silhouttes).abs().mean()
        reconstruction_err = (col_err + sil_err) / 2.
        
        KL_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

        loss = self.reconstruction_coef * reconstruction_err + self.KLD_coef * KL_divergence
        if self.perceptional_loss:
                # rendered_images, target_renders = rendered_images.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1), target_renders.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1)
                # rendered_silhouttes, target_silhouttes = rendered_silhouttes.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1), target_silhouttes.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1)
                rendered_silhouttes = rendered_silhouttes.expand(-1, 3, -1, -1, -1)
                feats_col = torch.cat((rendered_images, target_renders), 0)
                feats_sil = torch.cat((rendered_silhouttes, target_silhouttes), 0)
                p_col_loss = self.feature_extractor_col(feats_col).mean()
                p_sil_loss = self.feature_extractor_sil(feats_sil).mean()
                loss += self.p_loss_coef * (p_col_loss + p_sil_loss)
        loss = (loss / batch_size).mean()

        if self.clamp:
            loss = torch.clamp(loss, min=-self.clamp_threshold, max=self.clamp_threshold)
        print(loss, reconstruction_err, p_col_loss, p_sil_loss, KL_divergence)
        return loss


class VAELossGAN(nn.Module):
    def __init__(self, device, clamp, clamp_threshold,  adv_net_sil, adv_net_sil_optim, adv_net_col, adv_net_col_optim, KLD_coef=1., p_loss_coef=1.):
        super(VAELossGAN, self).__init__()
        self.KLD_coef = KLD_coef
        self.p_loss_coef = p_loss_coef

        self.clamp = clamp
        self.clamp_threshold = clamp_threshold

        self.adv_net_sil = adv_net_sil
        self.adv_net_col = adv_net_col
        self.adv_net_sil_optim = adv_net_sil_optim
        self.adv_net_col_optim = adv_net_col_optim
        self.criterion = nn.BCEWithLogitsLoss()

        self.device=device

    # question: how is the loss function using the mu and variance?
    def forward(self, mu, logvar, rendered_images, rendered_silhouttes, target_renders, target_silhouttes, batch_size):
        """gives the batch normalized Variational Error."""

        # rendered_images, target_renders = rendered_images.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1), target_renders.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1)
        # rendered_silhouttes, target_silhouttes = rendered_silhouttes.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1), target_silhouttes.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1)
        rendered_silhouttes = rendered_silhouttes.expand(-1, 3, -1, -1, -1)
        
        # GAN loss
        real_labels = torch.full(size=(rendered_images.shape[0], 1), fill_value=0.9, device=self.device)
        fake_labels = torch.full(size=(rendered_images.shape[0], 1), fill_value=0.1, device=self.device)


        labels = torch.cat((real_labels, fake_labels))
        preds_real_sil = self.adv_net_sil(target_silhouttes)
        preds_fake_sil = self.adv_net_sil(rendered_silhouttes.detach())   
        preds_sil = torch.cat((preds_real_sil, preds_fake_sil))
        loss_desc_sil = self.criterion(preds_sil, labels)

        # update descriptor
        self.adv_net_sil_optim.zero_grad()
        loss_desc_sil.backward()
        self.adv_net_sil_optim.step()

        preds_real_col = self.adv_net_col(target_renders)
        preds_fake_col = self.adv_net_col(rendered_images.detach())
        preds_col = torch.cat((preds_real_col, preds_fake_col))
        loss_desc_col = self.criterion(preds_col, labels)
        
        self.adv_net_col_optim.zero_grad()
        loss_desc_col.backward()
        self.adv_net_col_optim.step()

        # GAN loss loss
        real_labels = torch.ones(rendered_images.shape[0], 1, device=self.device)
        fake_preds_sil = self.adv_net_sil(rendered_silhouttes)
        fake_preds_col = self.adv_net_col(rendered_images)
        gen_sil_loss = self.criterion(fake_preds_sil, real_labels)
        gen_col_loss = self.criterion(fake_preds_col, real_labels)

        GAN_loss = gen_sil_loss + gen_col_loss
        
        # KL loss
        KL_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        
        # perceptional loss (need for generated model to be same as encoded
        feats_col = torch.cat((rendered_images, target_renders), 0)
        feats_sil = torch.cat((rendered_silhouttes, target_silhouttes), 0)
        p_col_loss = self.adv_net_col.perc_loss(feats_col)
        p_sil_loss = self.adv_net_sil.perc_loss(feats_sil)

        loss_gen = GAN_loss + self.KLD_coef * KL_divergence + self.p_loss_coef * (p_col_loss + p_sil_loss)
        loss_gen = (loss_gen / batch_size).mean()

        if self.clamp:
            loss_gen = torch.clamp(loss_gen, min=-self.clamp_threshold, max=self.clamp_threshold)
        
        return loss_gen