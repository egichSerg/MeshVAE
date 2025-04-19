from torch.utils.checkpoint import checkpoint_sequential

from tqdm import tqdm


def max_VRAM_optim_train_step(modules, optimizer, criterion, dataloader, device, scheduler):
    segments = len(modules)
    avg_loss = 0.
    for input_renders, label_renders in tqdm(dataloader):
        input_renders, label_renders = input_renders.to(device), label_renders.to(device)
        output_renders = checkpoint_sequential(modules, segments, input_renders)

        #make random renders, take random label renders, downsample label renders and THEN calculate loss
        
        loss = criterion(output_renders, label_renders)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not scheduler is None:
            scheduler.step()

        avg_loss += loss.item()

    avg_loss = avg_loss / len(dataloader)
    return avg_loss


def eval_step(model, criterion, dataloader, device):
    model.eval()
    avg_loss = 0.
    with torch.inference_mode():
        for input_renders, label_renders in tqdm(dataloader):
            input_renders, label_renders = input_renders.to(device), label_renders.to(device)
            output_renders = model(input_renders)

            #make random renders, take random label renders, downsample label renders and THEN calculate loss
            
            loss = criterion(output_renders, label_renders)

        avg_loss = avg_loss / len(dataloader)
    return avg_loss


def max_VRAM_optim_fit(model):
    