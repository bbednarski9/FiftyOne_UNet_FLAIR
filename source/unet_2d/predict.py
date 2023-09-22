import os
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from .utils.dice_score import dice_coeff
from .utils.dice_score import multiclass_dice_coeff

def apply_unet(model, dataloader, mask_save_pth: str = None):
    assert os.path.exists(mask_save_pth), f"save_masks: {mask_save_pth} does not exist."
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device)
    
    #mask_values = state_dict.pop('mask_values', [0, 1])

    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=False):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, image_pth, mask_true_pth = batch['image'], batch['mask'], batch['image_pth'], batch['mask_pth']
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = model(image)

            if model.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]. Got: {mask_true.min()} and {mask_true.max()}'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred.squeeze(), mask_true.squeeze(), reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < model.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            patient_slice = mask_true_pth[0].split("/")[-1].replace("_mask.npy","")
            patient = patient_slice.rsplit('_',1)[0]
            root_save = os.path.join(mask_save_pth, patient)
            if not os.path.exists(root_save):
                os.makedirs(root_save)

            np.save(os.path.join(root_save, patient_slice + '_mask.npy'), mask_pred.cpu())

    #net.train()
    return dice_score / max(num_val_batches, 1)