import torch
import datetime
import os

def train(model, optimizer, criterion, train_dataloader, validation_dataloader, 
            val_chk_freq=10, epochs=100, scheduler=None, device='cpu'):
    loss_history = {'train':[], 'val':[]}
    last_loss = 10 ** 9

    model.switch2forward()
    print('train mode start!!!!!!!!!!!!!!!!!!!!!!!!!!')

    for epoch in range(epochs):
        model.train()        
        total_loss = 0
        for iter, (_, train_img, train_labels) in enumerate(train_dataloader):
            score = model(train_img.to(device)) # might be [20]
            score = torch.log(score)

            optimizer.zero_grad()
            loss = criterion(score, train_labels.reshape(-1).to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss)

            if scheduler is not None:
                scheduler.step()

        total_loss /= len(train_dataloader)
        loss_history['train'].append(total_loss)
        print('====================================')
        print("train epoch %d, loss : %f "%(epoch + 1, total_loss))
        
        # get validation loss
        total_trainval_loss = validate(model, criterion, validation_dataloader, device=device)
        loss_history['val'].append(total_trainval_loss)
        print('++++++++++++++++++++++++++++++++++++')
        print("validation epoch %d, loss : %f "%(epoch + 1, total_trainval_loss))

        if (epoch + 1) % val_chk_freq == 0: # if loss_history['val'][-2] > loss_history['val'][-1]:
            if last_loss > total_trainval_loss:
                now = datetime.datetime.now()
                PATH = "checkpoint/model_%d_%d_%d_%d_%d" % (now.month, now.day, now.hour, now.minute, epoch + 1)
                torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'loss': total_trainval_loss
                            }, PATH)
                last_loss = total_trainval_loss

    print("Training End!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return loss_history


def validate(model, criterion, validation_dataloader, device='cpu'):
    model.eval()
    total_trainval_loss = 0
    for iter, (_, trainval_img, trainval_labels) in enumerate(validation_dataloader):
        score = model(trainval_img.to(device))
        score = torch.log(score)
        
        loss = criterion(score, trainval_labels.reshape(-1).to(device))
        total_trainval_loss += float(loss)

    total_trainval_loss /= len(validation_dataloader)
    return total_trainval_loss
