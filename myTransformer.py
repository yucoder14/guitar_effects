import torch 
import torch.nn as nn
import math

"""
    The Attention Layer 
"""
class AttentionLayer(nn.Module): 
    def __init__(
        self,
        model_dim, # d
        key_dim,   # d_k
        num_heads=1,
        dropout_rate=0.1
    ):
        super().__init__() 

        if num_heads < 1:
            raise ValueError("num_heads cannot be less than 1!")

        if model_dim % num_heads != 0: 
            raise ValueError("model_dim is not divisible by num_heads!")

        self.dropout = nn.Dropout(p=dropout_rate)
        self.model_dim = model_dim
        self.key_dim = key_dim
        self.value_dim = model_dim//num_heads 
        self.num_heads = num_heads

        # without batching considerations...
        #self.W_Q = nn.Parameter(torch.rand((num_heads, model_dim, key_dim))) 
        #self.W_K = nn.Parameter(torch.rand((num_heads, model_dim, key_dim))) 
        #self.W_V = nn.Parameter(torch.rand((num_heads, model_dim, self.value_dim))) 
        # self.W_O = nn.Parameter(torch.rand((model_dim, model_dim)))

        # using nn.Linear to automatically do batching  
        # concatenate heads vertically for broadcasting
        self.W_Q = nn.Linear(in_features=model_dim, out_features=num_heads * key_dim, bias=False)
        self.W_K = nn.Linear(in_features=model_dim, out_features=num_heads * key_dim, bias=False)
        self.W_V = nn.Linear(in_features=model_dim, out_features=num_heads * self.value_dim, bias=False)
        self.W_O = nn.Linear(in_features=num_heads * self.value_dim, out_features=model_dim, bias=False)
        
    def forward(self, X, H_enc=None, mask=None): # X has (batch_size, N, model_dim) dimensions
        K_input = H_enc if H_enc is not None else X
        V_input = H_enc if H_enc is not None else X

        batch_size = X.shape[0]
        N = X.shape[1]
        M = K_input.shape[1]

        # without batching considerations 
        #Q = X@self.W_Q # (num_heads, N, key_dim)
        #K = K_input@self.W_K # (num_heads, N, key_dim)
        #V = V_input@self.W_V # (num_heads, N, value_dim)

        Q = self.W_Q(X)       # (batch_size, N, num_heads * key_dim)   
        K = self.W_K(K_input) # (batch_size, M, num_heads * key_dim) 
        V = self.W_V(V_input) # (batch_size, M, num_heads * key_dim) 

        Q = Q.view(batch_size, N, self.num_heads, self.key_dim).transpose(1,2) 
        K = K.view(batch_size, M, self.num_heads, self.key_dim).transpose(1,2)
        V = V.view(batch_size, M, self.num_heads, self.value_dim).transpose(1,2)

        attention = Q@(K.mT) / math.sqrt(self.key_dim) # (batch_size, num_heads, N (queries), M (keys)) 
        attention = self.dropout(attention)
        if mask is not None:
            current_mask = mask[:N, :M] # I should consider lengths of N, M carefully 
            attention += current_mask 
            
        prob = nn.functional.softmax(attention, dim = -1) # dim -1 should be keys 
        values = prob@V #(batch_size, num_heads, N, value_dim)
        values_cat = values.transpose(1, 2).contiguous().view(batch_size, N, -1) # (batch_size, N, num_heads * value_dim)
        
        #cat_heads = torch.cat(heads.unbind(), dim=1) # (N, value_dim) each and concatenate the columns to form (N, model_dim)
        A = self.W_O(values_cat) # (N, model_dim)

        return A
        
class FeedForward(nn.Module):
    def __init__(
        self, 
        model_dim, 
        hidden_dim,
        dropout_rate=0.1              
    ):
        super().__init__()
        self.lin_1 = nn.Linear(in_features=model_dim, out_features=hidden_dim)
        self.lin_2 = nn.Linear(in_features=hidden_dim, out_features=model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, X):
        return self.dropout(self.lin_2(self.relu(self.lin_1(X))))

class LayerNorm(nn.Module): 
    def __init__(self, model_dim, epsilon=1e-6): 
        super().__init__()
        # gamma and beta is to normalize features 
        self.gamma = nn.Parameter(torch.ones(model_dim)) # initialize gammas to ones because if initialized randomly to 0, it's dead signal
        self.beta = nn.Parameter(torch.zeros(model_dim))
        self.eps = epsilon
        
    def forward(self, X): # (batch_size, N, model_dim)
        model_dim = X.shape[-1]
        mean = torch.mean(X, dim=-1, keepdims=True)
        std = torch.std(X, dim=-1, keepdims=True)
        X_hat = (X - mean) / (std + self.eps) # add epsilon for numerical stability
        layer_norm = self.gamma * X_hat + self.beta
        return layer_norm

class TransformerBlock(nn.Module): 
    def __init__(
        self, 
        N, 
        model_dim, 
        key_dim, 
        hidden_dim, 
        num_heads=1,
        cross_attention=False,
        attention_dropout_rate=0.1,
        ffn_dropout_rate=0.1, 
        block_dropout_rate=0.1
    ): 
        super().__init__() 
        self.N = N
        self.model_dim = model_dim
        self.key_dim = key_dim 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads
        self.cross_attention = cross_attention

        self.dropout = nn.Dropout(p=block_dropout_rate)
        self.attention_layer = AttentionLayer(model_dim=model_dim, key_dim=key_dim, num_heads=num_heads, dropout_rate=attention_dropout_rate)
        self.ffn = FeedForward(model_dim=model_dim, hidden_dim=hidden_dim, dropout_rate=ffn_dropout_rate)
        self.norm_1 = LayerNorm(model_dim=model_dim)
        self.norm_2 = LayerNorm(model_dim=model_dim)

        if self.cross_attention:
            self.cross_attention_layer = AttentionLayer(model_dim=model_dim, key_dim=key_dim, num_heads=num_heads)
            self.norm_3 = LayerNorm(model_dim=model_dim)

    def forward(self, X, H_enc=None, mask=None): 
        T_1 = self.norm_1(X) 
        T_2 = self.attention_layer(T_1, mask=mask)
        T_3 = T_2 + X
        if self.cross_attention: 
            if H_enc is None:
                raise ValueError("H_enc cannot be None if cross_attention is enabled!")
            T_a = self.norm_3(T_3) 
            T_b = self.cross_attention_layer(T_a, H_enc=H_enc)
            T_3 += T_b
        T_4 = self.norm_2(T_3)
        T_5 = self.ffn(T_4)
        H = T_5 + T_3 

        return self.dropout(H)

class TransformerStack(nn.Module): 
    def __init__(
        self, 
        N, 
        model_dim, 
        key_dim, 
        hidden_dim, 
        num_heads=1,
        cross_attention=False,
        num_stack=1,
        attention_dropout_rate=0.1,
        ffn_dropout_rate=0.1,
        block_dropout_rate=0.1
    ):
        super().__init__()

        if num_stack < 1: 
            raise ValueError("num_stack cannot be less than 1!")

        # using module list a
        self.blocks = nn.ModuleList([
            TransformerBlock(
                N, 
                model_dim, 
                key_dim, 
                hidden_dim, 
                num_heads, 
                cross_attention=cross_attention, 
                attention_dropout_rate=attention_dropout_rate, 
                ffn_dropout_rate=ffn_dropout_rate, 
                block_dropout_rate=block_dropout_rate
            ) 
             for _ in range(num_stack)
        ])
    def forward(self, X, H_enc=None, mask=None): 
        for block in self.blocks:
            X = block(X, H_enc, mask)
        return X