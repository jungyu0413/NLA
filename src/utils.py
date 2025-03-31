import torch
import cv2
import numpy as np
import random
import torch.nn.functional as F
import os

# save model
def save_classifier(model, epoch, args):
    path = os.path.join(args.output,f'{epoch}.pth')
    torch.save(model.state_dict(), path)
    print(f'save : {epoch} : {path}')

# generate folder
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# calculate accuracy      
class AccuraryLogger(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.reset(num_class)

    def reset(self, n):
        self.classwise_sum = np.zeros(n, dtype=float)
        self.classwise_count = np.zeros(n, dtype=float)
        self.total_sum = 0
        self.total_count = 0
    
    def update(self, predictions, labels):
        
        # Get number of images in current batch. 
        num_imgs = predictions.shape[0]
        
        # Store total values. 
        self.total_sum += np.sum((predictions == labels))
        self.total_count += num_imgs        
        
        # Store class-wise values. 
        for i in range(self.classwise_sum.shape[0]):  # 7 class: 0 ~ 6
            self.classwise_sum[i] += np.sum((predictions == i) * (labels == i)).astype(float)  # Correct
            self.classwise_count[i] += np.sum((labels == i)).astype(float)  # Wrong

    def final_score(self):
        
        # Calculate classwise accuracy. 
        classwise_acc = self.classwise_sum / self.classwise_count
        for idx,cnt in enumerate(self.classwise_count):
            if cnt == 0:
                classwise_acc[idx] = 1
        # Calculate total mean accuracy. 
        total_acc = self.total_sum / self.total_count
        
        return classwise_acc, total_acc



class AccuraryLogger_top2(object):
    """Computes and stores the average and current value"""

    def __init__(self, num_class):
        self.reset(num_class)

    def reset(self, n):
        self.classwise_sum = np.zeros(n, dtype=float)
        self.classwise_count = np.zeros(n, dtype=float)
        self.total_sum = 0
        self.total_count = 0
        
        self.top2_classwise_sum = np.zeros(n, dtype=float)  # 각 클래스별 top-2 정확도의 총 합
        self.top2_classwise_count = np.zeros(n, dtype=float) 
        self.top2_sum = 0

        
    def update(self, predictions, predictions_top2, labels):
        
        # Get number of images in current batch. 
        num_imgs = predictions.shape[0]
        
        # Store total values. 
        self.total_sum += np.sum((predictions == labels))
        self.total_count += num_imgs        
        
        top2_indices = np.argsort(predictions_top2, axis=1)[:, ::-1][:, :2]
        top2_correct = np.sum(np.any(top2_indices == np.expand_dims(labels, axis=1), axis=1))
        self.top2_sum += top2_correct
        
        # Store class-wise values. 
        for i in range(self.classwise_sum.shape[0]):  # 7 class: 0 ~ 6
            self.classwise_sum[i] += np.sum((predictions == i) * (labels == i)).astype(float)  # Correct
            self.classwise_count[i] += np.sum((labels == i)).astype(float)  # Wrong
            
        for i in range(self.classwise_sum.shape[0]):  # 7 class: 0 ~ 6
            correct_top2 = np.sum((predictions_top2[:, :2] == i) * (labels[:, None] == i), axis=1)
            self.top2_classwise_sum[i] += np.sum(correct_top2)
            self.top2_classwise_count[i] += np.sum(labels == i)
    
    
    def final_score(self):
        
        # Calculate classwise accuracy. 
        classwise_acc = self.classwise_sum / (self.classwise_count + 1e-8)
        for idx,cnt in enumerate(self.classwise_count):
            if cnt == 0:
                classwise_acc[idx] = 1
        # Calculate total mean accuracy. 
        total_acc = self.total_sum / self.total_count
        top2_acc = self.top2_sum / self.total_count
        
        return classwise_acc, total_acc, top2_acc


# img augmentation
def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

