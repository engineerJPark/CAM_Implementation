import torch
import datetime
import os
import torchvision.transforms as transforms

def train(model, optimizer, criterion, train_dataloader, validation_dataloader, 
            save_freq=10, epochs=100, scheduler=None, device='cpu'):    
    
    loss_history = {'train':[], 'val':[]}
    last_loss = 10 ** 9

    # model.switch2forward()
    print('train mode start!!!!!!!!!!!!!!!!!!!!!!!!!!')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for iter, (train_img, train_labels) in enumerate(train_dataloader):
            train_img = train_img.squeeze()
            train_img = torch.stack([train_img, torch.flip(train_img, [-1])])
            train_labels = torch.stack((train_labels, train_labels))
        
            score = model(normalization(train_img).to(device)) # might be [2, 20]
            optimizer.zero_grad()
            loss = criterion(score, train_labels.reshape(-1, 20).to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss)

            if scheduler is not None:
                scheduler.step()

        total_loss /= len(train_dataloader)
        loss_history['train'].append(total_loss)
        
        # get validation loss
        total_trainval_loss = validate(model, criterion, validation_dataloader, device=device)
        loss_history['val'].append(total_trainval_loss)
        print('====================================')
        print("epoch %d, train loss : %f "%(epoch + 1, total_loss), "validation loss : %f "%(total_trainval_loss))

        if (epoch + 1) % save_freq == 0 and last_loss > total_trainval_loss:
            now = datetime.datetime.now()
            PATH = "./checkpoint/model_%d_%d_%d_%d_%d" % (now.month, now.day, now.hour, now.minute, epoch + 1)
            torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'loss': total_trainval_loss
                        }, PATH)
            last_loss = total_trainval_loss

    print("Training End!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    if PATH is not None:
        return loss_history, PATH
    else:
        return loss_history, None


def validate(model, criterion, validation_dataloader, device='cpu'):
    model.eval()
    total_trainval_loss = 0
    for iter, (trainval_img, trainval_labels) in enumerate(validation_dataloader):
        trainval_img = trainval_img.squeeze()
        trainval_img = torch.stack([trainval_img, torch.flip(trainval_img, [-1])])
        trainval_labels = torch.stack((trainval_labels, trainval_labels))
        score = model(normalization(trainval_img).to(device))
        
        loss = criterion(score, trainval_labels.reshape(-1, 20).to(device))
        total_trainval_loss += float(loss)

    total_trainval_loss /= len(validation_dataloader)
    return total_trainval_loss

def normalization(img):
    tf = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [1., 1., 1.], inplace=False) # std = [0.229, 0.224, 0.225]
    img = tf(img)
    return img


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1