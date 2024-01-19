import torch
import torch.nn as nn
from network.models import model_selection
import torch.optim as optim
from torch.optim import lr_scheduler
from dataset.transform import xception_default_data_transforms
import os
import pickle
import mydata

def train(model_path):
    losses_acc_dict={}
    name = 'FFDD'
    epoches = 20
    batch_size=512
    output_path = '.\\output\\fs_1201'
    model_name='deepburger.pkl'
    real_data, fake_data = mydata.read_data()
    train_data = real_data[:int(len(real_data)*0.8)] + fake_data[:int(len(fake_data)*0.8)]
    val_data = real_data[int(len(real_data)*0.8):] + fake_data[int(len(fake_data)*0.8):]
    train_dataset=mydata.MyDataset(train_data,transform=xception_default_data_transforms['train'])
    val_dataset=mydata.MyDataset(val_data,transform=xception_default_data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False,num_workers=8)
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path)) 
    for i,(name,param) in enumerate(model.named_parameters()):
        param.requires_grad=False
        if i == 145:
            break
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0e-07, betas=(0.9, 0.999), eps=1e-08)
    early_stopping_epochs=3
    best_loss=float('inf')
    early_stop_counter=0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model = nn.DataParallel(model)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch+1, epoches))
        print('-'*10)
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        for (image, labels) in train_loader:
            iter_loss = 0.0
            iter_corrects = 0.0
            image = image.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration % 20):
                print('iteration {} train loss: {} Acc: {}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {} Acc: {}'.format(epoch_loss, epoch_acc))

        model.eval()
        with torch.no_grad():
            for (image, labels) in val_loader:
                image = image.cuda()
                labels = labels.cuda()
                outputs = model(image)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_val_loss = val_loss / val_dataset_size
            epoch_val_acc = val_corrects / val_dataset_size
            print('epoch val loss: {} Acc: {}'.format(epoch_val_loss, epoch_val_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
		#if not (epoch % 40):
        torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
        losses_acc_dict[str(epoch)]=(epoch_loss,epoch_acc,epoch_val_loss,epoch_val_acc)
        with open('./losses_acc_dict2.txt','wb') as f:
            pickle.dump(losses_acc_dict,f)
        if epoch_val_loss > best_loss:
            early_stop_counter+=1
        else:
            best_loss = epoch_val_loss
            early_stop_counter=0

        if early_stop_counter >= early_stopping_epochs:
            print("Early Stopping!!!")
            break
    print('Best val Acc: {}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.module.state_dict(), os.path.join(output_path, "trf_best.pkl"))

if __name__=='__main__':
    train('C:\\Users\\administrator\\torch\\models\\deepfake_c0_xception.pkl')