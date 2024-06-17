import torch.nn as nn
import torch
import torch.nn.functional as F


class EmbModel(nn.Module):
    '''
    Class used either on the training or inference stage of self-supervised learning model
    '''

    def __init__(self, base_encoder, args):
        super().__init__()
        self.enc = base_encoder(pretrained=args['pretrained'])
        self.feature_dim = self.enc.fc.in_features
        self.projection_dim = args['projection_dim'] 
        self.proj_hidden = 512

        # remove final fully connected layer of the backbone
        self.enc.fc = nn.Identity()  

        if args['store_embeddings']:
            self.emb_memory = torch.zeros(args['num_train'], args['projection_dim'], 
                                          requires_grad=False, device=args['device'])


        # standard simclr projector
        self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.proj_hidden),
                                       nn.ReLU(),
                                       nn.Linear(self.proj_hidden, self.projection_dim)) 

    def update_memory(self, inds, x):
        m = 0.9
        with torch.no_grad():
            self.emb_memory[inds] = m*self.emb_memory[inds] + (1.0-m)*F.normalize(x.detach().clone(), dim=1, p=2)
            self.emb_memory[inds] = F.normalize(self.emb_memory[inds], dim=1, p=2)        

    def forward(self, x, only_feats=False, context=None):
        op = {}
        op['feat'] = self.enc(x) 
        if not only_feats:        
            op['emb'] = self.projector(op['feat'])

        return op


class AdapterMLP(nn.Module):
    '''
    MLP Network for low-shot adaptation (trained on top of frozen features)
    '''
    def __init__(self,num_classes,input_size,hidden_size):
        super(AdapterMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        out = self.mlp(x)
        return out