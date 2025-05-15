
import os
import logging

logger = logging.getLogger(__name__)

import numpy as np

import torch

from ..logger import Logger


def print_train():
    ''' Most important function '''
    logger.info('------------------------------------')
    logger.info('------------------------------------')
    logger.info('')
    logger.info('         ..oo0  ...ooOO00           ')
    logger.info('        ..     ...             !!!  ')
    logger.info('       ..     ...      o       \o/  ')
    logger.info('   Y  ..     /III\    /L ---    n   ')
    logger.info('  ||__II_____|\_/| ___/_\__ ___/_\__')
    logger.info('  [[____\_/__|/_\|-|______|-|______|')
    logger.info(' //0 ()() ()() 0   00    00 00    00')
    logger.info('')
    logger.info('------------------------------------')
    logger.info('------------------------------------')
    
def get_last_checkpoint(directory, max_epochs=10000):
    ''' Get last checkpoint of a model from a directory '''
    checkpoint = None
    for i in range(max_epochs):
        temp = os.path.join(directory, f'epoch_{i}')
        if not os.path.exists(temp):
            return checkpoint
        checkpoint = temp

def get_device(device):
    ''' Convenient function to set up hardward '''
    if device.lower() == 'cpu':
        device = torch.device('cpu')
    elif 'cuda' in device.lower():
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            logging.warning('No GPU available. Use CPU instead.')
            device = torch.device('cpu')
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory *= 1e-9 # convert bytes to Gb
        logger.info('- Use device: {}'.format(torch.cuda.get_device_name(device)))
        logger.info('- Total memory: {:.4f} GB'.format(total_memory))
    else:
        logger.info('- Use device: CPU')
    return device


def train(train_loader, model, criterion, optimizer, lr_scheduler, 
          val_loader=None, max_epochs=1, logger=None, device='cpu'):
    """Train network with CompositePSDLoss that requires witness input"""
    
    if logger is None:
        logger = Logger(outdir='outdir', label='run', metrics=['loss'])
    
    num_batches = len(train_loader)
    print_train()
    
    for epoch in range(max_epochs):
        train_loss = 0.
        model.train()
        
        for i_batch, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            data = data.to(device)
            witness = data[:, :-1, :]  # all channels except last
            target = data[:, -1, :]    # last channel is target
            
            pred = model(witness)
            loss = criterion(pred, target, witness)
            loss.backward()
            
            # Clip gradients after backward but before step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if criterion.reduction == 'mean':
                train_loss += loss.item() * len(data)
            else:
                train_loss += loss.item()
        
        # Validation
        val_loss = 0.
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for i_batch, (data, _) in enumerate(val_loader):
                    data = data.to(device)
                    witness = data[:, :-1, :]
                    target = data[:, -1, :]

                    pred = model(witness)
                    loss = criterion(pred, target, witness)
                    
                    if criterion.reduction == 'mean':
                        val_loss += loss.item() * len(data)
                    else:
                        val_loss += loss.item()
        
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset) if val_loader is not None else 1.0  # 避免除以0
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        logger.update_metric(train_loss, val_loss, 'loss', epoch, 
                             num_batches, num_batches)
        logger.display_status(epoch, max_epochs, num_batches, num_batches,
                              train_loss, val_loss, 'loss')
        logger.log_metric()
        logger.save_model(model, epoch)


def evaluate(dataloader, model, criterion=None, device='cpu'):
    """Evaluate model and return predictions and (optional) loss"""
    model.eval()
    eval_loss = 0.
    prediction = []

    with torch.no_grad():
        for i_batch, (data, _) in enumerate(dataloader):
            data = data.to(device)
            witness = data[:, :-1, :]
            target = data[:, -1, :]

            pred = model(witness)
            prediction.append(pred.cpu().numpy())
            
            if criterion is not None:
                loss = criterion(pred, target, witness)
                if criterion.reduction == 'mean':
                    eval_loss += loss.item() * len(data)
                else:
                    eval_loss += loss.item()

    prediction = np.concatenate(prediction)
    eval_loss /= len(dataloader.dataset) if criterion is not None else 1.0
    
    return (prediction, eval_loss) if criterion is not None else prediction