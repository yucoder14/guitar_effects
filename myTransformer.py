import torch 
import torch.nn as nn
import dac
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
        dropout_rate=0.1, 
        enc_out_dim=None,
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
        self.enc_out_dim = enc_out_dim

        # without batching considerations...
        #self.W_Q = nn.Parameter(torch.rand((num_heads, model_dim, key_dim))) 
        #self.W_K = nn.Parameter(torch.rand((num_heads, model_dim, key_dim))) 
        #self.W_V = nn.Parameter(torch.rand((num_heads, model_dim, self.value_dim))) 
        # self.W_O = nn.Parameter(torch.rand((model_dim, model_dim)))

        # using nn.Linear to automatically do batching  
        # concatenate heads vertically for broadcasting
        self.W_Q = nn.Linear(in_features=self.model_dim, out_features=self.num_heads * self.key_dim, bias=False)
        # in features should allow for different dimensions! because H_enc may have different dimensions!
        enc_out_dim = self.enc_out_dim if enc_out_dim is not None else self.model_dim
        self.W_K = nn.Linear(in_features=enc_out_dim, out_features=self.num_heads * self.key_dim, bias=False)
        self.W_V = nn.Linear(in_features=enc_out_dim, out_features=self.num_heads * self.value_dim, bias=False)
        self.W_O = nn.Linear(in_features=self.num_heads * self.value_dim, out_features=self.model_dim, bias=False)
        
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
        model_dim, 
        key_dim, 
        hidden_dim, 
        num_heads=1,
        enc_out_dim=None,
        attention_dropout_rate=0.1,
        ffn_dropout_rate=0.1, 
        block_dropout_rate=0.1
    ): 
        super().__init__() 
        self.model_dim = model_dim
        self.key_dim = key_dim 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads
        self.enc_out_dim = enc_out_dim

        self.dropout = nn.Dropout(p=block_dropout_rate)
        self.attention_layer = AttentionLayer(
            model_dim=model_dim, 
            key_dim=key_dim, 
            num_heads=num_heads, 
            dropout_rate=attention_dropout_rate
        )
        self.ffn = FeedForward(model_dim=model_dim, hidden_dim=hidden_dim, dropout_rate=ffn_dropout_rate)
        self.norm_1 = LayerNorm(model_dim=model_dim)
        self.norm_2 = LayerNorm(model_dim=model_dim)

        if self.enc_out_dim is not None:
            self.cross_attention_layer = AttentionLayer(
                model_dim=model_dim, 
                key_dim=key_dim, 
                num_heads=num_heads, 
                dropout_rate=attention_dropout_rate,
                enc_out_dim=self.enc_out_dim
            )
            self.norm_3 = LayerNorm(model_dim=model_dim)

    def forward(self, X, H_enc=None, mask=None): 
        T_1 = self.norm_1(X) 
        T_2 = self.attention_layer(T_1, mask=mask)
        T_3 = T_2 + X
        
        if self.enc_out_dim is not None: 
            if H_enc is None:
                raise ValueError("H_enc cannot be None if cross_attention is enabled!")
            T_a = self.norm_3(T_3) 
            T_b = self.cross_attention_layer(T_a, H_enc=H_enc)
            # T_3 += T_b --> this messes up autograd because it's an in-place operation that overwrites T_3 
            T_3 = T_3 + T_b # this is fine 
            
        T_4 = self.norm_2(T_3)
        T_5 = self.ffn(T_4)
        H = T_5 + T_3 

        return self.dropout(H)

class TransformerStack(nn.Module): 
    def __init__(
        self, 
        model_dim,  # d_model
        key_dim,    # d_key
        hidden_dim, 
        num_heads=1,
        enc_out_dim=None,
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
                model_dim, 
                key_dim, 
                hidden_dim, 
                num_heads, 
                enc_out_dim=enc_out_dim, 
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

# from https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
# i don't completely understand positional encoding yet, but I have built the intuition that 
# it is analogous to how binary numbers encode numbers; smaller bits flips more frequently 
# than larger bits; this is modeled by the sinusodial waves 
# it also takes advantage of linearity of trigonometric addition formulas, which supposedly 
# helps the model to figure out relative positioning...
# https://medium.com/thedeephub/positional-encoding-explained-a-deep-dive-into-transformer-pe-65cfe8cfe10b 
class PositionalEncoding(nn.Module):

    def __init__(self, model_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, 1, model_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        X = X + self.pe[:X.size(0)]
        return self.dropout(X)

"""
    Culmination of everything thus far. 
    Using Dac as an encoder to derive the hidden representation of audio; using discrete code
    Then, I proceed with a decoder in a canonical fashion
"""
class MyModel(nn.Module): 
    def __init__(
        self, 
        dac, 
        num_vocab, 
        model_dim, 
        key_dim, 
        enc_out_dim, 
        ffn_hidden_dim, 
        num_heads=1,
        num_stack=1, 
        N = 32,
        M = 128,
        padding_idx=0
    ):
        super().__init__() 

        self.embed = nn.Embedding(
            num_embeddings=num_vocab + 1, 
            embedding_dim=model_dim, 
            padding_idx=padding_idx
        ) 
        
        mask = torch.tensor(
            [[0 if i>= j else -torch.inf for j in range(max(N,M))] for i in range(max(N,M))]
        ) 
        self.register_buffer('mask', mask)
        
        self.pos = PositionalEncoding(model_dim=model_dim)
        
        self.dac = dac
        for param in self.dac.parameters():
            param.requires_grad = False
            
        self.decoder = TransformerStack(
            model_dim=model_dim, 
            key_dim=key_dim, 
            hidden_dim=ffn_hidden_dim, 
            num_heads=num_heads, 
            enc_out_dim=enc_out_dim,
            num_stack=num_stack
        )
        
        self.unembed = nn.Linear(model_dim, num_vocab + 1)

    def forward(self, source, target): 
        """
            Returns logits of the target, which can then be passed to nn.CrossEntropyLoss
        """
        # encoder 
        _, H, _, _, _ = self.dac.encode(source) 
        H = H.transpose(1, 2) * 1.0
        
        # decoder
        X = self.embed(target)
        X = self.pos(X) 
        X = self.decoder(X, H, self.mask) 
        X = self.unembed(X)
        
        return X