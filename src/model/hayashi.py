import torch
import torch.nn as nn
import numpy as np
from src.model import lupts 
from .. import base
from . import modelutils

class LinearRegression(torch.nn.Module): 
    def __init__(self, dim_input, dim_output=1): 
        super(LinearRegression, self).__init__()
        self.model = torch.nn.Linear(dim_input, dim_output)
    
    def forward(self, X): 
        y_hat = self.model(X)
        return y_hat

class HayashiModel(base.Model): 
    
    def __init__(self, args={}, model_args={}, train_args={}, distill_seq=False): 
        self.args  = args
        self.model_args = model_args
        self.train_args = train_args
        self.distill_seq = distill_seq
        
        intercept = args.get('intercept_last_step') if 'intercept_last_step' in args else True
        if not self.distill_seq: 
            self.teacher_model = modelutils.linear_model(self.args, model_args=self.model_args, intercept=intercept)
        else: 
            self.teacher_model = lupts.LUPTS()
        self.student_model = None

    def hayashi_loss(self, y_hat, y_hard, y_soft, lambda_): 
        y_hard = y_hard.squeeze()
        y_soft = y_soft.squeeze()
        y_hat  = y_hat.squeeze()
        loss = 0.5*((1-lambda_)*torch.nn.functional.mse_loss(y_hat,y_hard) + lambda_*torch.nn.functional.mse_loss(y_hat,y_soft)) # replace mse
        return loss 

    def set_train_args(self, train_args): 
        # used to set the train args after a hyperparameter search
        assert 'lambda' in train_args.keys() 
        assert 'epochs' in train_args.keys()
        assert 'lr' in train_args.keys()
        self.train_args = train_args 

    def fit(self, X : np.array, y_hard : np.array): 
        
        # generate y_soft 
        Xbase  = X[:,0,:]
        if self.distill_seq:
            #Pretty sure this should be Xpriv instead of X
            #self.teacher_model.fit(X,y_hard)
            #y_soft = self.teacher_model.predict(X)
            Xpriv = X[:,1:,:]
            self.teacher_model.fit(Xpriv,y_hard)
            y_soft = self.teacher_model.predict(Xpriv)
        else: 
            Xpriv  = np.reshape(X[:,1:,:],(X[:,1:,:].shape[0],-1))
            teacher_fitted = self.teacher_model.fit(Xpriv,y_hard)
            y_soft = teacher_fitted.predict(Xpriv)
        
        # instantiate student model 
        self.student_model = LinearRegression(Xbase.shape[1],1)
        if torch.cuda.is_available(): 
            self.student_model.cuda()
        
        lambda_ = self.train_args['lambda'] if ('lambda' in self.train_args) else 0.5
        epochs  = self.train_args['epochs'] if ('epochs' in self.train_args) else 40
        lr      = self.train_args['lr'] if ('lr' in self.train_args) else 1e0
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)

        #Somehow introduce hyper tuning for lambda

        for epoch in range(epochs):
            # Converting inputs and labels to Variable
            if torch.cuda.is_available():
                inputs = torch.from_numpy(Xbase).type(torch.FloatTensor).cuda()
                Y_hard = torch.from_numpy(y_hard).type(torch.FloatTensor).cuda()
                Y_soft = torch.from_numpy(y_soft).type(torch.FloatTensor).cuda()
            else:
                inputs = torch.from_numpy(Xbase).type(torch.FloatTensor)
                Y_hard = torch.from_numpy(y_hard).type(torch.FloatTensor)
                Y_soft = torch.from_numpy(y_soft).type(torch.FloatTensor)

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # get output from the model, given the inputs
            y_hat = self.student_model(inputs)

            # get loss for the predicted output
            loss = self.hayashi_loss(y_hat, Y_hard, Y_soft, lambda_)
            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()
            
            '''
            if epoch % 50 == 0: 
                print('epoch {}, loss {}'.format(epoch, loss.item()))
            '''

    def predict(self, X : np.array) -> np.array:
        Xbase = X[:,0,:]
        if torch.cuda.is_available(): 
            inputs = torch.from_numpy(Xbase).type(torch.FloatTensor).cuda()
        else: 
            inputs = torch.from_numpy(Xbase).type(torch.FloatTensor)
        
        Y_hat = self.student_model(inputs).squeeze()
        return Y_hat.detach().cpu().numpy()