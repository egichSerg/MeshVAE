import torch
import torch.nn as nn

from model_collection.VAE_loss.vgg19 import VGG19

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

        self.feature_extractor = VGG19().to(device)

    # question: how is the loss function using the mu and variance?
    def forward(self, mu, logvar, rendered_images, rendered_silhouttes, target_renders, target_silhouttes, batch_size):
        """gives the batch normalized Variational Error."""

        col_err = self.criterion(target_renders, rendered_images).abs().mean()
        sil_err = self.criterion(target_silhouttes, rendered_silhouttes).abs().mean()
        reconstruction_err = (col_err + sil_err) / 2.
        
        KL_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

        loss = self.reconstruction_coef * reconstruction_err + self.KLD_coef * KL_divergence
        if self.perceptional_loss:
                rendered_images, target_renders = rendered_images.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1), target_renders.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1)
                rendered_silhouttes, target_silhouttes = rendered_silhouttes.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1), target_silhouttes.squeeze().unsqueeze(1).expand(-1, 3, -1, -1, -1)
                feats_col = torch.cat((rendered_images, target_renders), 0)
                feats_sil = torch.cat((rendered_silhouttes, target_silhouttes), 0)
                p_col_loss = self.feature_extractor(feats_col)
                p_sil_loss = self.feature_extractor(feats_sil)
                loss += self.p_loss_coef * (p_col_loss + p_sil_loss)
        loss = (loss / batch_size).mean()

        if self.clamp:
            loss = torch.clamp(loss, min=-self.clamp_threshold, max=self.clamp_threshold)
        
        return loss