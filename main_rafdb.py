import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "3" 
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import wandb
from src.dataset import NLA_Rafdb
from src.model import NLA_r18
from src.loss import * 
from src.utils import *
from src.resnet import *
from src.test import *
from src.train import *
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="rafdb", type=str, choices=['rafdb', 'Ferplus', 'AffecNet'], help="experiment dataset")
parser.add_argument('--exp_name', default="NAW_NLA", type=str, choices=['NAW_NLA', 'NAW_CAM', 'NAW_L1', 'L1', 'CAM', 'JSD'], help="training strategy")
parser.add_argument('--exp_ver', default="temp", type=str, help="experiment version and date")
parser.add_argument('--dataset_path', type=str, default='/workspace/NLA/imgs/RAF-DB', help='raf_dataset_path')
parser.add_argument('--label_path', type=str, default='/workspace/NLA/imgs/list_patition_label.txt', help='label_path')
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--feature_embedding', type=int, default=512)
parser.add_argument('--output', default="/workspace/NLA/AAAI", type=str, help="output dir")
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--lr', type=float, default=0.0001, help='lr')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--save_freq', type=int, default=5, help='save frequency')  
# other setting
parser.add_argument('--noise', type=bool, default=False, help='learning from noise label')  
parser.add_argument('--noise_name', type=str, default="10_1", help='noise percentage_random seed')  
parser.add_argument('--imbalanced', type=bool, default=False, help='learning from imbalanced label')
parser.add_argument('--imbalanced_name', type=str, default="20_256", help='imbalance factor_random seed')  
parser.add_argument('--seed', type=int, default=0) # 11111135
parser.add_argument('--lam_a', type=float, default=0.5)
parser.add_argument('--lam_b', type=float, default=0.5)
parser.add_argument('--lam_c', type=float, default=0.5)
parser.add_argument('--slope', type=float, default=-15)
parser.add_argument('--t_lambda', type=float, default=1)
parser.add_argument('--sch_bool', type=bool, default=True)
parser.add_argument('--mu_x_t', type=float, default=0.5) 
parser.add_argument('--mu_y_t', type=float, default=0.5) 
parser.add_argument('--mu_x_f', type=float, default=0.30) 
parser.add_argument('--mu_y_f', type=float, default=0.15) 
parser.add_argument('--t_std_major', type=float, default=0.8) 
parser.add_argument('--t_std_ratio', type=float, default=2) 
parser.add_argument('--f_std_major', type=float, default=0.91) 
parser.add_argument('--f_std_ratio', type=float, default=6) 
args = parser.parse_args()

        
def main():    
    setup_seed(args.seed)
    args.con = 0
    args.exp_name = args.exp_name
    args.output = os.path.join(args.output, args.dataset, args.exp_name)
    args.max_acc = 0
    args.max_acc_mean = 0
    args.save_cnt = 0
    createDirectory(args.output)
    hyper_setting = ['true', str(args.mu_x_t) ,str(args.mu_y_t), str(args.t_std_major), str(args.t_std_ratio), 
    'false',str(args.mu_y_t) ,str(args.mu_y_f), str(args.f_std_major), str(args.f_std_ratio), 'original']
    wandb.init(project='project1', name=args.exp_name+'_'+"_".join(hyper_setting))
    wandb_args = {
            "backbone":'ResNet18',
            "batch_size": args.batch_size,
            'num_gpu': 'A6000*1'
            }
    wandb.config.update(wandb_args)   
    
    

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.577, 0.4494, 0.4001],
                            std=[0.2628, 0.2395, 0.2383]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomErasing(scale=(0.02, 0.3)) ]) # 0.25


    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.577, 0.4494, 0.4001],
                            std=[0.2628, 0.2395, 0.2383])])
    train_dataset = NLA_Rafdb(args, phase='train', transform=train_transforms)
    test_dataset = NLA_Rafdb(args, phase='test', transform=eval_transforms)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.workers,
                                            pin_memory=True)
        
    
    
    
    model = NLA_r18(args)
    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters() , lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    
    for idx, i in enumerate(range(1, args.epochs + 1)):
        train_acc, train_loss = train(args, idx, model, train_loader, optimizer, scheduler, device)
        wandb.log({"Train Acc@1": train_acc,
        "Train Loss" : train_loss,
        }, step=idx)
        
        print(f'Train: [{idx}/{args.epochs + 1}]\t'
        f'Train Acc@1 {train_acc:.4f}\t'
        f'Train Loss {train_loss:.3f}\t')

        
        acc_metric, test_acc, test_loss = test(model, test_loader, device)
        classwise_acc, total_acc, top2_acc = acc_metric.final_score()
        wandb.log({"Test Acc@1": total_acc,
        "Test Loss" : test_loss,
        "Test Mean Acc" : np.mean(classwise_acc),
        "Top 2 Acc" : top2_acc,
        'Neutral' : classwise_acc[6],
        'Happiness' : classwise_acc[3],
        'Sadness' : classwise_acc[4],
        'Surprise' : classwise_acc[0],
        'Fear' : classwise_acc[1],
        'Disgust' : classwise_acc[2],
        'Anger' : classwise_acc[5],
        }, step=idx)
        
        print(f'Test: [{idx}/{args.epochs + 1}]\t'
        f'Test Acc@1 {test_acc:.4f}\t'
        f'Test Mean Acc {np.mean(classwise_acc):.4f}\t'
        f'Test Loss {test_loss:.3f}\t')
        print(f'class acc : {classwise_acc}')
        
        if args.max_acc_mean < np.mean(classwise_acc):
            save_classifier(model, 'best_mean', args)
            args.max_acc_mean = np.mean(classwise_acc)
            
        if args.max_acc < test_acc:
            save_classifier(model, 'best', args)
            args.max_acc = test_acc
            
        # if  idx % 5 == 0:
        #     save_classifier(model, str(idx), args)
            
if __name__ == '__main__':
    main()
