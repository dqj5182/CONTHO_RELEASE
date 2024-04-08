import torch
import torch.nn as nn
from torch.nn import functional as F

from funcs_utils import init_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=2048+3, nhead=1, dim_feedforward=2048+3, kdim=256+3+3, vdim=256+3+3, dropout=0.0, activation="gelu"):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim_feedforward, nhead, dropout=dropout, kdim=kdim, vdim=vdim)
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.linear1 = nn.Linear(dim_feedforward, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_feedforward)
        self.norm1 = nn.LayerNorm(dim_feedforward)
        self.norm2 = nn.LayerNorm(dim_feedforward)
        self.activation = F.gelu

    def forward(self, src_q, src_k, src_v):
        src_q, src_k, src_v = src_q.permute(1,0,2), src_k.permute(1,0,2), src_v.permute(1,0,2)
        src = src_q
        src2, _ = self.cross_attn(src_q, src_k, src_v)

        src = src + self.dropout(src2)
        src = self.norm(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.permute(1,0,2)


class ContactFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.human_fc_in = nn.Linear(2048, 256)
        self.object_fc_in = nn.Linear(2048, 256)
        self.CA_Transformer_human, self.CA_Transformer_object = [], []
        self.num_layers = 4
        self.dim = 256+3

        for i in range(self.num_layers):
            transformer_layer = TransformerEncoderLayer(d_model=self.dim, nhead=1, dim_feedforward=self.dim, kdim=self.dim, vdim=self.dim)
            self.CA_Transformer_human.append(transformer_layer)
            transformer_layer = TransformerEncoderLayer(d_model=self.dim, nhead=1, dim_feedforward=self.dim, kdim=self.dim, vdim=self.dim)
            self.CA_Transformer_object.append(transformer_layer)
        self.CA_Transformer_human = nn.ModuleList(self.CA_Transformer_human)
        self.CA_Transformer_object = nn.ModuleList(self.CA_Transformer_object)
        self.fc_out_human = nn.Linear(self.dim, 1)
        self.fc_out_object = nn.Linear(self.dim, 1)

    def init_weights(self):
        self.apply(init_weights)
        
    def forward(self, human_kps, object_kps, human_tokens, object_tokens):
        human_tokens, object_tokens = self.human_fc_in(human_tokens), self.object_fc_in(object_tokens)
        human_tokens, object_tokens = torch.cat((human_tokens, human_kps), -1), torch.cat((object_tokens, object_kps), -1)
        contact_human_tokens, contact_object_tokens = human_tokens.clone(), object_tokens.clone()

        for i in range(self.num_layers):
            contact_human_tokens_ = self.CA_Transformer_human[i](contact_human_tokens, contact_object_tokens, contact_object_tokens)
            contact_object_tokens_ = self.CA_Transformer_object[i](contact_object_tokens, contact_human_tokens, contact_human_tokens)
            contact_human_tokens, contact_object_tokens = contact_human_tokens_, contact_object_tokens_
            
        h_contacts = self.fc_out_human(contact_human_tokens)[:,:,0].sigmoid()
        o_contacts = self.fc_out_object(contact_object_tokens)[:,:,0].sigmoid()
        return human_tokens, object_tokens, h_contacts, o_contacts


class CRFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.SA_Transformer_human, self.SA_Transformer_object = [], []
        self.CA_Transformer_human, self.CA_Transformer_object = [], []
        self.Transformer_human, self.Transformer_object = [], []
        self.num_layers = 4
        self.dim = 256+3
        
        for i in range(self.num_layers):
            transformer_layer = TransformerEncoderLayer(d_model=self.dim, nhead=1, dim_feedforward=self.dim, kdim=self.dim, vdim=self.dim)
            self.SA_Transformer_human.append(transformer_layer)
            transformer_layer = TransformerEncoderLayer(d_model=self.dim, nhead=1, dim_feedforward=self.dim, kdim=self.dim, vdim=self.dim)
            self.SA_Transformer_object.append(transformer_layer)
            transformer_layer = TransformerEncoderLayer(d_model=self.dim, nhead=1, dim_feedforward=self.dim, kdim=self.dim, vdim=self.dim)
            self.CA_Transformer_human.append(transformer_layer)
            transformer_layer = TransformerEncoderLayer(d_model=self.dim, nhead=1, dim_feedforward=self.dim, kdim=self.dim, vdim=self.dim)
            self.CA_Transformer_object.append(transformer_layer)
            transformer_layer = TransformerEncoderLayer(d_model=self.dim, nhead=1, dim_feedforward=self.dim, kdim=self.dim, vdim=self.dim)
            self.Transformer_human.append(transformer_layer)
            transformer_layer = TransformerEncoderLayer(d_model=self.dim, nhead=1, dim_feedforward=self.dim, kdim=self.dim, vdim=self.dim)
            self.Transformer_object.append(transformer_layer)
            
        self.SA_Transformer_human = nn.ModuleList(self.SA_Transformer_human)
        self.SA_Transformer_object = nn.ModuleList(self.SA_Transformer_object)
        self.CA_Transformer_human = nn.ModuleList(self.CA_Transformer_human)
        self.CA_Transformer_object = nn.ModuleList(self.CA_Transformer_object)
        self.Transformer_human = nn.ModuleList(self.Transformer_human)
        self.Transformer_object = nn.ModuleList(self.Transformer_object)

        self.fc_out_human = nn.Linear(self.dim, 3)
        self.fc_out_object = nn.Linear(self.dim, 3)

    def init_weights(self):
        self.apply(init_weights)
        
    def forward(self, human_tokens, object_tokens, h_contacts, o_contacts):
        human_contacts, obj_contacts = (h_contacts.detach() > 0.5).bool(), (o_contacts.detach() > 0.5).bool()
        c_human_tokens, c_object_tokens = human_tokens.clone(), object_tokens.clone()
        c_human_tokens[~human_contacts] = 0; c_object_tokens[~obj_contacts] = 0
        
        for i in range(self.num_layers):
            human_tokens = self.SA_Transformer_human[i](human_tokens, human_tokens, human_tokens)
            object_tokens = self.SA_Transformer_object[i](object_tokens, object_tokens, object_tokens)
            
        for i in range(self.num_layers):
            c_human_tokens_ = self.CA_Transformer_human[i](c_human_tokens, c_object_tokens, c_object_tokens)
            c_object_tokens_ = self.CA_Transformer_object[i](c_object_tokens, c_human_tokens, c_human_tokens)
            c_human_tokens, c_object_tokens = c_human_tokens_, c_object_tokens_
        
        human_tokens = c_human_tokens + human_tokens
        object_tokens = c_object_tokens + object_tokens
        
        for i in range(self.num_layers):
            human_tokens = self.Transformer_human[i](human_tokens, human_tokens, human_tokens)
            object_tokens = self.Transformer_object[i](object_tokens, object_tokens, object_tokens)
            
        human_kps, object_kps = self.fc_out_human(human_tokens), self.fc_out_object(object_tokens)
        return human_kps, object_kps