import torch
import datetime

def train(model, optimizer, criterion, train_dataloader, validation_dataloader, 
            scheduler=None, epochs=100, device='cpu'):
    loss_history = []
    last_loss = 10 ** 9

    model.switch2forward()
    model.train()
    print('train mode start!!!!!!!!!!!!!!!!!!!!!!!!!!')

    for epoch in range(epochs):
        
        total_loss = 0
        for iter, (train_img, train_labels) in enumerate(train_dataloader):
            score = model(train_img.to(device)) # might be [20]

            optimizer.zero_grad()
            loss = criterion(score, train_labels.reshape(-1).to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        total_loss /= len(train_dataloader)

        print('====================================')
        print("epoch %d, loss : %f "%(epoch + 1, total_loss))
        loss_history.append(loss.item())

        if scheduler is not None:
            scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            total_trainval_loss = 0
            for iter, (trainval_img, trainval_labels) in enumerate(validation_dataloader):
                score = model(trainval_img.to(device))
                loss = criterion(score, train_labels.reshape(-1).to(device))
                total_trainval_loss += float(loss)

            total_trainval_loss /= len(train_dataloader)

            if last_loss > total_trainval_loss:
                now = datetime.datetime.now()
                PATH = "checkpoint/model_%d_%d_%d_%d_%d" % (now.month, now.day, now.hour, now.minute, epoch + 1)
                torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            #   'optimizer_state_dict': optimizer.state_dict(),
                            #   'scheduler_state_dict': scheduler.state_dict(),
                            #   'loss': total_trainval_loss
                            }, PATH)
                last_loss = total_trainval_loss

    print("Training End!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
