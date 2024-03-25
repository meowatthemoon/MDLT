import torch
import torch.nn as nn

import math

class InputEmbeddings(nn.Module):
    def __init__(self, input_size : int, d_model : int):
        super().__init__()
        self.d_model : int = d_model
        self.embedding = nn.Linear(input_size, self.d_model) # Changed, instead of Embed layer -> Linear, and instead of vocab size -> input size which can be state_size or action_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model : int, max_ep_len : int, dropout : float):
        super().__init__()
        self.d_model : int = d_model
        self.max_ep_len : int = max_ep_len
        self.dropout = nn.Dropout(dropout)

        # Create matrix of shape (seq_len, d_model)
        self.embed_timestep = nn.Embedding(max_ep_len, d_model)

    def forward(self, state, timestep):
        x = state + self.embed_timestep(timestep)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps : float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.ones(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # No changes, had initially thought -2 since I thought seq_len would be second to last, be after embedding it should be last aswell right?
        std = x.std(dim = -1, keepdim = True) # Same here
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model : int, d_ff : int, dropout : float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model : int, n_heads : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model is not divisible by n_heads"
        
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Module):
        d_k = query.shape[-1]

        # (Batch, n_head, seq_len, d_k) -> # (Batch, n_head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # Replace all values where mask is 0 with -inf for softmax (softmax basically replaces by 0)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, n_head, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, n_head, d_k) -->  (Batch, n_head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query = query, key = key, value = value, mask = mask, dropout = self.dropout)

        # (Batch, n_head, seq_len, d_k) --> (Batch, seq_len, n_head, d_k) --> (Batch, seq_len, d_model) 
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) 
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout : float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttention, feed_forward_block : FeedForward, dropout : float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout = dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)  
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttention, cross_attention_block : MultiHeadAttention, feed_forward_block : FeedForward, dropout : float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout = dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)  
        return self.norm(x) 

class ProjectionLayer(nn.Module):
    def __init__(self, d_model : int, action_size : int):
        super().__init__()
        self.proj = nn.Linear(d_model, action_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, action_size)
        return torch.tanh(self.proj(x))#, dim = -1) # Changes, instead of log_softmax, we do tanh

class Transformer(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder, state_embed : InputEmbeddings, action_embed : InputEmbeddings, timestep_embed : PositionalEmbedding, projection_layer : ProjectionLayer):
        super().__init__()
        self.state_embed = state_embed
        self.action_embed = action_embed
        self.timestep_embed = timestep_embed # Changes, we only use one pos encoder

        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def encode(self, state, timestep, src_mask):
        state = self.state_embed(state)
        state = self.timestep_embed(state, timestep)
        x =  self.encoder(state, src_mask)
        return x

    def decode(self, encoder_output, src_mask, action, timestep, tgt_mask):
        action = self.action_embed(action)
        action = self.timestep_embed(action, timestep)
        return self.decoder(x = action, encoder_output = encoder_output, src_mask = src_mask, tgt_mask = tgt_mask) 

    def project(self, x):
        return self.projection_layer(x)

def build_transfomer(state_size : int, action_size : int, max_ep_len : int, d_model : int = 512, n_layers : int = 6, n_heads : int = 8, dropout : float = 0.1, d_ff : int = 2048) -> Transformer:
    # Create embedding layers
    state_embed = InputEmbeddings(input_size = state_size, d_model = d_model)
    action_embed = InputEmbeddings(input_size = action_size, d_model = d_model)

    # Pos encoding
    timestep_embed = PositionalEmbedding(d_model = d_model, max_ep_len = max_ep_len, dropout = dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(n_layers):
        self_attention_block = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        feed_forward_block = FeedForward(d_model = d_model, d_ff = d_ff, dropout = dropout)
        encoder_block = EncoderBlock(self_attention_block = self_attention_block, feed_forward_block = feed_forward_block, dropout = dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(n_layers):
        self_attention_block = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        cross_attention_block = MultiHeadAttention(d_model = d_model, n_heads = n_heads, dropout = dropout)
        feed_forward_block = FeedForward(d_model = d_model, d_ff = d_ff, dropout = dropout)
        decoder_block = DecoderBlock(self_attention_block = self_attention_block, cross_attention_block = cross_attention_block, feed_forward_block = feed_forward_block, dropout = dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model = d_model, action_size = action_size)

    # Create the transformer
    transformer = Transformer(encoder = encoder, decoder = decoder, state_embed = state_embed, action_embed = action_embed, timestep_embed = timestep_embed, projection_layer = projection_layer)

    # Initialize weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

if __name__ == '__main__':
    build_transfomer(state_size = 438, action_size = 4, seq_len = 20)
