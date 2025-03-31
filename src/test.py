import torch
from src.utils import *
from tqdm import tqdm
import torch.nn as nn

def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0
        acc = AccuraryLogger_top2(7)

        for image, labels, _ in tqdm(test_loader):
            image = image.to(device)
            labels = labels.to(device)


            outputs = model(image)


            loss = nn.CrossEntropyLoss()(outputs, labels).detach()

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num

            running_loss += loss.item()
            data_num += outputs.size(0)
            _, predicts = torch.max(outputs, 1)
            preds = predicts.detach().cpu().numpy() 
            targets = labels.detach().cpu().numpy()
            preds_top2 = outputs.detach().cpu().numpy()
            acc.update(preds, preds_top2, targets)
            
        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)
        
    return acc, test_acc, running_loss