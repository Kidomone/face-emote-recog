import os
import time
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import f1_score
import torch
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler, DataLoader


import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)


from utils.Custom_models import Resnet_Custom
from utils.Datasets import AffectNet_dataset

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

affectnet_labels_names =   [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
  ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

from torch.profiler import profile, record_function, ProfilerActivity #----------


def train_epoch(model, optimizer, criteria, dataloader, epoch=None):
    global device
    global scaler
    model.train()

    for inputs, labels  in dataloader:
        optimizer.zero_grad()

        outputs = model(inputs.to(device, non_blocking=True))
        loss = criteria(outputs, labels.to(device, non_blocking=True))

        loss.backward()
        optimizer.step()

        # metrics.update(outputs, labels, loss=loss.item())

    return model


def test_epoch(model, criteria, dataloader, epoch=None, to_show=False):
    global device
    model.eval()

    num = 0
    y_true = []
    y_pred = []
    loss_all = 0
    with torch.no_grad():
        for inputs, labels  in dataloader:
            
            outputs = model(inputs.to(device, non_blocking=True))
            loss = criteria(outputs, labels.to(device, non_blocking=True))

            loss_all += loss
            preds = torch.argmax(outputs, dim=1)
            y_true.append(labels)
            y_pred.append(preds)

            # metrics.update(outputs=outputs, labels=labels, loss=loss.item(), is_test=True)

    print(f"Epo test  {epoch}:   Loss: {loss:0.3f}")


    from sklearn.metrics import classification_report

    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    target_names = affectnet_labels_names

    print(classification_report(
        y_true, 
        y_pred, 
        target_names=target_names,
        zero_division=np.nan
    ))
    print("")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    if to_show:
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=target_names).plot(xticks_rotation='vertical')
        plt.show()
   
    return model, f1_score(y_true, y_pred, average='macro')





def add_param_group(optimizer, model, layer_name, lr=1e-4, weight_decay=1e-5):
    new_params = [
        param for name, param in model.named_parameters()
        if (layer_name in name and param.requires_grad and not any(param is pg_param for pg in optimizer.param_groups for pg_param in pg['params']))
    ]
    if new_params:
        optimizer.add_param_group({
            'params': new_params,
            'lr': lr,
            'weight_decay': weight_decay
        })


def save_model(model, epo='test', to_state=False):
    name = f"ML/models/Resnet_Custom_{str(epo)}.pth"
    if to_state:
        torch.save(model.state_dict(), name)
    else:
        torch.save(model, name)




if __name__ == "__main__":

    num_epo = 20
    start_epo = 1
    batch_size = 128

    torch.cuda.empty_cache()
    model = Resnet_Custom(output_shape=len(affectnet_labels_names))
    model_name = "Resnet_AffectNet"
    model.to(device)
    
    transforms = T.Compose([
        T.Grayscale(1),
        T.RandomResizedCrop((84, 70), scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(7),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        #T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transforms = T.Compose([
        T.Grayscale(1),
        T.CenterCrop((84, 70)),
        #T.ToTensor(),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.5], std=[0.5])
    ])




    dataset_train = AffectNet_dataset(transform=transforms, cache_to_ram=True)
    dataset_test = AffectNet_dataset(transform=test_transforms, is_test=True)

    num_workers = min(8, max(1, (os.cpu_count() or 4) - 2)) 
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=2)

    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW([
    {'params': model.conv1.parameters(), 'lr': 5e-5},
    {'params': model.fc.parameters(),    'lr': 1e-4}
    ], weight_decay=1e-4)

    # scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 30, 40], gamma=0.4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-7)
    best_f1 = 0
    model, best_f1 = test_epoch(model, criteria, dataloader_test, 0)
    for epoch in range(start_epo, num_epo+1):
        try:
            if epoch == 3:
                for name, param in model.named_parameters():
                    if 'layer1' in name:
                        param.requires_grad = True
                add_param_group(optimizer, model, 'layer1', lr=1e-4)

            if epoch == 5:
                for name, param in model.named_parameters():
                    if 'layer2' in name:
                        param.requires_grad = True
                add_param_group(optimizer, model, 'layer2', lr=1e-4)

            if epoch == 7:
                for name, param in model.named_parameters():
                    if 'layer3' in name:
                        param.requires_grad = True
                add_param_group(optimizer, model, 'layer3', lr=1e-4)

    
            model = train_epoch(model, optimizer, criteria, dataloader_train, epoch)
            #print("")
            #if epoch%3==0:
            model, f1_macro = test_epoch(model, criteria, dataloader_test, epoch)

            if f1_macro > best_f1:
                best_f1 = f1_macro
                save_model(model, 'best_f1', to_state=True)

        
            scheduler.step()

        except KeyboardInterrupt:
            print('Stopping by manual command')
            break

    model = test_epoch(model, criteria, dataloader_test, epoch, to_show=True)

    print('saving model')
    save_model(model, 'final')
    print('saved final model')    