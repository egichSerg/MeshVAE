import torch, os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

def train_step(model, optimizer, criterion, dataloader, accuracy_fn, device, scheduler, train_sils=False):
    model.train()
    avg_loss, avg_acc = 0., 0.
    for signals, labels in tqdm(dataloader):
        if train_sils:
            _, signals = signals.to(device).split([1, 1], dim = 1)
        else:
            signals, _ = signals.to(device).split([1, 1], dim = 1)
            
        signals = signals.squeeze()
        signals, labels = signals.to(device), labels.to(device)
        y_logits = model(signals).squeeze()
        y_logits = y_logits.unsqueeze(0) if len(y_logits.shape) < 2 else y_logits
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = criterion(y_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not scheduler is None:
            scheduler.step()

        avg_loss += loss.item()
        avg_acc += accuracy_fn(y_pred, labels)

    avg_loss, avg_acc = avg_loss / len(dataloader), avg_acc / len(dataloader)
    return (avg_loss, avg_acc)

def eval_step(model, criterion, dataloader, accuracy_fn, device, train_sils=False):
    model.eval()
    avg_loss, avg_acc = 0., 0.
    with torch.inference_mode():
        for signals, labels in tqdm(dataloader):
            if train_sils:
                _, signals = signals.to(device).split([1, 1], dim = 1)
            else:
                signals, _ = signals.to(device).split([1, 1], dim = 1)
                
            signals = signals.squeeze()
            signals, labels = signals.to(device), labels.to(device)
            y_logits = model(signals).squeeze()
            y_logits = y_logits.unsqueeze(0) if len(y_logits.shape) < 2 else y_logits
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

            avg_loss += criterion(y_logits, labels).item()
            avg_acc += accuracy_fn(y_pred, labels)

        avg_loss, avg_acc = avg_loss / len(dataloader), avg_acc / len(dataloader)
    return (avg_loss, avg_acc)

def fit(model, train_launch, optimizer, criterion, train_dl, test_dl, accuracy_fn, device, epochs, scheduler = None,
        save_weights=True, weights_save_dir=None, weights_save_mode='loss', early_stopping=True, 
        early_stopping_mode='loss', restore_best_weights=True, early_stopping_tolerance = 5, train_step_func=train_step, eval_step_func=eval_step,
        train_sils=False):
    '''fit model. Early stopping modes: \'acc\' - early stop by accuracy \'loss\' - early stop by loss '''
    
    MODEL_SAVE_PATH = weights_save_dir
    if save_weights:
        if weights_save_mode != 'loss' and weights_save_mode != 'acc':
            raise Exception('Unknown weights save mode. Possible modes:\n\'acc\' - save by accuracy;\n\'loss\' - save by loss\nDefault mode: \'loss\'')
        min_save_loss = 1000000.
        max_save_acc = 0.
        MODEL_SAVE_PATH = weights_save_dir / model.__class__.__name__ / f'{train_launch:02d}'
        MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    
    if early_stopping:
        if early_stopping_mode != 'loss' and early_stopping_mode != 'acc':
            raise Exception('Unknown early stopping mode. Possible modes:\n\'acc\' - stop by accuracy;\n\'loss\' - stop by loss\nDefault mode: \'loss\'')
        cache_weights_path = Path('.') / 'cached_weights.pt'
        tolerance = early_stopping_tolerance
        min_loss = 1000000.
        max_acc = 0.
    
    # the training cycle
    model.to(device)
    for epoch in range(epochs):
        print(f'==========Epoch {epoch+1}==========')
        print('Training...')
        train_loss, train_acc = train_step_func(model, optimizer, criterion, train_dl, accuracy_fn, device, scheduler, train_sils=train_sils)        
        
        print('Validating...')
        test_loss, test_acc = eval_step_func(model, criterion, test_dl, accuracy_fn, device, train_sils=train_sils)

        print(f'Train loss: {train_loss:.02f} | Train accuracy: {train_acc:.02f} | Test loss: {test_loss:.02f} | Test accuracy: {test_acc:.02f}')
        
        
        if save_weights:
            if weights_save_mode == 'loss':
                if min_save_loss > test_loss:
                    min_save_loss = test_loss
                    torch.save(model.state_dict(), MODEL_SAVE_PATH / 'best.pt')
                    print(f'Saved weights to {MODEL_SAVE_PATH / "best.pt"}')
                else:
                    torch.save(model.state_dict(), MODEL_SAVE_PATH / 'last.pt')
                    print(f'Saved weights to {MODEL_SAVE_PATH / "last.pt"}')
            else:
                if max_save_acc < test_acc:
                    max_save_acc = test_acc
                    torch.save(model.state_dict(), MODEL_SAVE_PATH / 'best.pt')
                    print(f'Saved weights to {MODEL_SAVE_PATH / "best.pt"}')
                else:
                    torch.save(model.state_dict(), MODEL_SAVE_PATH / 'last.pt')
                    print(f'Saved weights to {MODEL_SAVE_PATH / "last.pt" }')
        
        
        if early_stopping:
            if early_stopping_mode == 'loss':
                if min_loss > test_loss:
                    min_loss = test_loss
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
                        return
            else:
                if max_acc < test_acc:
                    max_acc = test_acc
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
                        return