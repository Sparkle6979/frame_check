# from turtle import forward
# import torch
import torch.nn as nn

# def  Linear_Model(nn):
#     def __init__(self,)


class DCN_Net(nn.Module):
    def __init__(self,output_size):
        super(DCN_Net,self).__init__()
        self.output_size = output_size
        self.feature_layer = nn.Sequential(
            
            nn.Conv2d(3,16,(3,3),(1,1),(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.Conv2d(16,16,(3,3),(1,1),(1,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4,4),4),


            nn.Conv2d(16,32,(3,3),(1,1),(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,(3,3),(1,1),(1,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4,4),4),
    

            nn.AdaptiveAvgPool2d(output_size=(7,7))
        )

        self.classify_layer = nn.Sequential(
            nn.Linear(7*7*32,output_size)
        )

    
    def forward(self,x):
        output = self.feature_layer(x)

        output = output.reshape(output.shape[0],-1)
        output = self.classify_layer(output)
        return output
