import torch
import torch.nn as nn
import torch.nn.functional as F

class MYDFX(nn.Module):
    def __init__(HP_gpu):
        super(MYDFX, self).__init__()
        print('build MYDFX model...')
        self.gpu=HP_gpu
        self.batch_size=16
        self.sentenceLenth=100
        self.embedding_dim=256
        self.POStag_dim=51
        self.category_dim=19
        self.label_num=3
        self.dropout=0.3

        # 声明CNN
        self.in_channels=1
        self.out_channels=6
        
        self.window_size1=3
        self.window_size2=4
        self.window_size3=5

        self.conv_dim1=self.sentenceLenth-self.window_size1+1
        self.conv_dim2=self.sentenceLenth-self.window_size2+1
        self.conv_dim3=self.sentenceLenth-self.window_size3+1
        self.conv_dim = self.out_channels * 3

        # 声明LSTM
        self.bilstm_flag=True
        self.hidden_dim = 300
        self.lstm_layer = 1
        if self.bilstm_flag:
            lstm_hidden= self.hidden_dim//2
        else:lstm_hidden= self.hidden_dim
        
        self.droplstm1 = nn.Dropout(self.dropout)
        self.droplstm2 = nn.Dropout(self.dropout)
        self.droplstm3 = nn.Dropout(self.dropout)
        self.droplstm4 = nn.Dropout(self.dropout)
        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)
        self.drop3 = nn.Dropout(self.dropout)
        self.drop4 = nn.Dropout(self.dropout)
        self.drop5 = nn.Dropout(self.dropout)
        self.drop6 = nn.Dropout(self.dropout)
        self.drop7 = nn.Dropout(self.dropout)
        self.drop8 = nn.Dropout(self.dropout)
        self.drop9 = nn.Dropout(self.dropout)
        self.drop10 = nn.Dropout(self.dropout)

        self.conv1=nn.Conv2d(self.in_channels, self.out_channels,(self.window_size1,self.embedding_dim))
        self.conv2=nn.Conv2d(self.in_channels, self.out_channels,(self.window_size2,self.embedding_dim))
        self.conv3=nn.Conv2d(self.in_channels, self.out_channels,(self.window_size3,self.embedding_dim))

        self.batchNorm1=nn.BatchNorm2d(self.out_channels)
        self.batchNorm2=nn.BatchNorm2d(self.out_channels)
        self.batchNorm3=nn.BatchNorm2d(self.out_channels)

        self.lstm1=nn.LSTM(self.embedding_dim+self.POStag_dim,lstm_hidden,num_layers=self.lstm_layer,
                           batch_first=True, bidirectional=self.bilstm_flag)

        self.lstm2=nn.LSTM(self.hidden_dim,lstm_hidden,num_layers=self.lstm_layer,
                           batch_first=True, bidirectional=self.bilstm_flag)

        self.lstm3=nn.LSTM(self.hidden_dim*2,lstm_hidden,num_layers=self.lstm_layer,
                           batch_first=True, bidirectional=self.bilstm_flag)

        self.lstm4=nn.LSTM(self.hidden_dim*3,lstm_hidden,num_layers=self.lstm_layer,
                           batch_first=True, bidirectional=self.bilstm_flag)

        self.fc1=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)
        self.fc2=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)
        self.fc3=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)
        self.fc4=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)
        self.fc5=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)
        self.fc6=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)
        self.fc7=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)
        self.fc8=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)
        self.fc9=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)
        self.fc10=nn.Linear(self.hidden_dim+self.conv_dim+self.category_dim, self.label_num)

        if self.gpu:
            self.droplstm1=self.droplstm1.cuda()
            self.droplstm2=self.droplstm2.cuda()
            self.droplstm3=self.droplstm3.cuda()
            self.droplstm4=self.droplstm4.cuda()
            
            self.drop1=self.drop1.cuda()
            self.drop2=self.drop2.cuda()
            self.drop3=self.drop3.cuda()
            self.drop4=self.drop4.cuda()
            self.drop5=self.drop5.cuda()
            self.drop6=self.drop6.cuda()
            self.drop7=self.drop7.cuda()
            self.drop8=self.drop8.cuda()
            self.drop9=self.drop9.cuda()
            self.drop10=self.dro10p.cuda()
            
            self.conv1=self.conv1.cuda()
            self.conv2=self.conv2.cuda()
            self.conv3=self.conv3.cuda()
            
            self.batchNorm1=self.batchNorm1.cuda()
            self.batchNorm2=self.batchNorm2.cuda()
            self.batchNorm3=self.batchNorm3.cuda()
            
            self.lstm1=self.lstm1.cuda()
            self.lstm2=self.lstm2.cuda()
            self.lstm3=self.lstm3.cuda()
            self.lstm4=self.lstm4.cuda()
            
            self.fc1=self.fc1.cuda()
            self.fc2=self.fc2.cuda()
            self.fc3=self.fc3.cuda()
            self.fc4=self.fc4.cuda()
            self.fc5=self.fc5.cuda()
            self.fc6=self.fc6.cuda()
            self.fc7=self.fc7.cuda()
            self.fc8=self.fc8.cuda()
            self.fc9=self.fc9.cuda()
            self.fc10=self.fc10.cuda()

    def forward(self, trainMat, trainCate, trainTag):
        #提取情感特征
        x=trainMat.view(self.batch_size,1,self.sentenceLenth,self.embedding_dim)#

        x1=F.relu(self.conv1(x))
        x1=self.batchNorm1(x1)
        x1=x1.view(self.batch_size,self.out_channels,self.conv_dim1)
        x1=F.max_pool2d(x1,(1,self.conv_dim1))
        x1=x1.view(self.batch_size, self.out_channels)

        x2=F.relu(self.conv2(x))
        x2=self.batchNorm2(x2)
        x2=x2.view(self.batch_size,self.out_channels,self.conv_dim2)
        x2=F.max_pool2d(x2,(1,self.conv_dim2))
        x2=x2.view(self.batch_size, self.out_channels)

        x3=F.relu(self.conv3(x))
        x3=self.batchNorm3(x3)
        x3=x3.view(self.batch_size,self.out_channels,self.conv_dim3)
        x3=F.max_pool2d(x3,(1,self.conv_dim3))
        x3=x3.view(self.batch_size, self.out_channels)

        x=torch.cat((x1,x2,x3),1)

        #标签特征
        y=trainCate.view(self.batch_size, self.category_dim)

        #上下文依赖特征
        z=torch.cat((trainMat,trainTag),2)

        h10=torch.randn(2,self.batch_size,150).cuda()
        c10=torch.randn(2,self.batch_size,150).cuda()
        z1,h1n=self.lstm1(z, (h10,c10))
        z1 = self.droplstm1(z1)

        
        h20=torch.randn(2,self.batch_size,150).cuda()
        c20=torch.randn(2,self.batch_size,150).cuda()
        z2,h2n=self.lstm2(z1, (h20,c20))
        z2 = self.droplstm2(z2)

        z12=torch.cat((z1,z2),2)
        h30=torch.randn(2,self.batch_size,150).cuda()
        c30=torch.randn(2,self.batch_size,150).cuda()
        z3,h3n=self.lstm3(z12, (h30,c30))
        z3 = self.droplstm3(z3)

        z123=torch.cat((z1,z2,z3),2)
        h40=torch.randn(2,self.batch_size,150).cuda()
        c40=torch.randn(2,self.batch_size,150).cuda()
        z4,h4n=self.lstm4(z123, (h40,c40))
        z4 = self.droplstm4(z4)

        z5=z4[:,0,:]

        #合并特征
        xyz=torch.cat((x,y,z5),1)
        xyz=xyz.view(self.batch_size,-1)

        output1=F.softmax(self.fc1(xyz),dim=1)
        output2=F.softmax(self.fc2(xyz),dim=1)
        output3=F.softmax(self.fc3(xyz),dim=1)
        output4=F.softmax(self.fc4(xyz),dim=1)
        output5=F.softmax(self.fc5(xyz),dim=1)
        output6=F.softmax(self.fc6(xyz),dim=1)
        output7=F.softmax(self.fc7(xyz),dim=1)
        output8=F.softmax(self.fc8(xyz),dim=1)
        output9=F.softmax(self.fc9(xyz),dim=1)
        output10=F.softmax(self.fc10(xyz),dim=1)

        return output1,output2,output3,output4,output5,output6,output7,output8,output9,output10
