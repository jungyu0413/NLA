import torch
from src.utils import *
from src.loss import *
from tqdm import tqdm
import torch.nn as nn



def train(args, idx, model, train_loader, optimizer, scheduler, device):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    acc = AccuraryLogger_top2(7)
    model.to(device)
    model.train()    
    # if 'GA' in args.exp_name:
    args.eps = exponential_scheduler(idx, args.epochs, args.slope, args.sch_bool)
        
    for image, labels, image2 in tqdm(train_loader):
        image = image.to(device)
        image2 = image2.to(device)
        labels = labels.to(device)
            
        output  = model(image)
        output2 = model(image2)
        # Loss
        cross_loss = nn.CrossEntropyLoss(reduction='none')(output, labels)
        t_major = args.t_std_major**2 + (args.t_std_major/args.t_std_ratio)**2
        t_minor = -args.t_std_major**2 + (args.t_std_major/args.t_std_ratio)**2
        
        f_major = args.f_std_major**2 + (args.f_std_major/args.f_std_ratio)**2
        f_minor = args.f_std_major**2 - (args.f_std_major/args.f_std_ratio)**2


        loss_naw = NAW(args, mu_x_t=args.mu_x_t, mu_y_t=args.mu_y_t, mu_x_f = args.mu_x_f, mu_y_f = args.mu_y_f, f_minor=f_minor, f_major=f_major, t_minor=t_minor, t_major=t_major, t_lambda=args.t_lambda)(output, labels)
        

        loss_jsd = args.lam_c * jensen_shannon_divergence(output, output2)
        
        loss = (args.lam_a * cross_loss.mean() + args.lam_b * loss_naw.mean()) + loss_jsd
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        iter_cnt += 1
        _, predicts = torch.max(output, 1)
        
        preds = predicts.detach().cpu().numpy() 
        targets = labels.detach().cpu().numpy()
        
        preds_top2 = output.detach().cpu().numpy()
        acc.update(preds, preds_top2, targets)
        
        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num
        running_loss += loss.item()
    
    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(train_loader.dataset.__len__())
    return acc, running_loss
        
        