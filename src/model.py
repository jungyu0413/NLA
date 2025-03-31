import pickle
from src.resnet import *
from src.resnet18 import *



        
        

class NLA_r18(nn.Module):

    def __init__(self, args):
        super(NLA_r18, self).__init__()
        Resnet18 = resnet18()
        cp = torch.load('/workspace/eac/Erasing-Attention-Consistency/src/resnet18_msceleb.pth')
        Resnet18.load_state_dict(cp['state_dict'])
        self.embedding = args.feature_embedding
        self.num_classes = args.num_classes
        self.features = nn.Sequential(*list(Resnet18.children())[:-2])  
        self.features2 = nn.Sequential(*list(Resnet18.children())[-2:-1])  
        self.fc = nn.Linear(self.embedding, self.num_classes)  
        self.embed = nn.Linear(self.embedding, 1024)
                    
    def forward(self, x):        
        x = self.features(x)
        #### 1, 2048, 7, 7
        feature = self.features2(x)
        #### 1, 2048, 1, 1   
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)

        return output
    

        
class NLA_r50(nn.Module):

    def __init__(self, args):
        super(NLA_r50, self).__init__()
        Resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        with open('/resnet50.pth', 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
            Resnet50.load_state_dict(weights)
        self.embedding = args.feature_embedding
        self.num_classes = args.num_classes
        
        self.features = nn.Sequential(*list(Resnet50.children())[:-2])  
        self.features2 = nn.Sequential(*list(Resnet50.children())[-2:-1])  
        self.fc = nn.Linear(self.embedding, self.num_classes)  

                    
    def forward(self, x):        
        x = self.features(x)
        #### 1, 2048, 7, 7
        feature = self.features2(x)
        #### 1, 2048, 1, 1   
        feature = feature.view(feature.size(0), -1)
        output = self.fc(feature)
            
        return output
        
        



