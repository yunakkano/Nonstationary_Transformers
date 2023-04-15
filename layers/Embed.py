import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        #print('TokenEmbedding x.permute(0,1,2) = ', x.permute(0, 2, 1).size())
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    detach() is used in the forward method of the FixedEmbedding class to prevent gradients 
    from being propagated through the embedding layer during training. 
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        # torch.arange() to create a tensor of integers from 0 to c_in, 
        # and then unsqueeze() is used to add a new dimension to make it a column vector. 
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        The detach() method is used to create a new tensor that shares the same underlying data as the original tensor 
        but is detached from the computation graph. When a tensor is detached, it prevents any gradients from being computed 
        or propagated back through it.
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4 # minutes data is grouped into 4 categories; 00-16 15-30 30-45 45-60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't': # t option means 'minutes' :  to_offset('t') -> <Minute>
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        Assuming that x has a shape of (batch_size, seq_length, 5), where batch_size is the size of the batch, 
        and seq_length is the length of each sequence in the batch, then the shape of minute_x, hour_x, weekday_x, day_x, 
        and month_x would be (batch_size, seq_length, d_model) since d_model is the output dimension of the Embedding layers.

        Refer data_provider.data_loader.Dataset.__read_data__() where datetime index is preprocessed to fit with the Embedding layer's input.
        """
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        """ 
        Return tensor would have a shape of (batch_size, seq_length, d_model).
        This type of operation is commonly used in neural networks that process sequential data, 
        such as recurrent neural networks (RNNs) and transformers, where each element in the sequence is represented 
        by a vector in a high-dimensional space, and these vectors are combined to form a representation of the entire sequence.
        """
        return hour_x + weekday_x + day_x + month_x + minute_x


class TemporalEmbeddingHighFreq(nn.Module):
    def __init__(self, d_model, seq_len, embed_type='fixed', freq='ns'):
        super(TemporalEmbeddingHighFreq, self).__init__()

        minute_size = 4
        hour_size = 8
        weekday_size = 7
        day_size = 4
        month_size = 4
        self.d_model = d_model
        self.seq_len = seq_len

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

        # Linear layer to process continuous seconds feature
        # self.seconds_linear = nn.Linear(seq_len, d_model)

        # To make the tensor of shape B x S x d_model from second_x tensor, prepare duplicator tensor
        # self.duplicator = torch.ones(1, 1, d_model).to(device='mps:0')

    def forward(self, x):
        second_x = x[:, :, 5].float() / 60.0 # Extract continuous(float) feature of seconds ( nanoseconds).
        x_discrete = x[:, :, :5].long() # Extract discrete features such as month day weekday hour or minute
        minute_x = self.minute_embed(x_discrete[:, :, 4])
        hour_x = self.hour_embed(x_discrete[:, :, 3])
        weekday_x = self.weekday_embed(x_discrete[:, :, 2])
        day_x = self.day_embed(x_discrete[:, :, 1])
        month_x = self.month_embed(x_discrete[:, :, 0])
        #print('TemporalEmbeddingHighFreq month_x = ', day_x.size())
        #print('TemporalEmbeddingHighFreq month_x = ', month_x.size())
        #print('TemporalEmbeddingHighFreq second_x = ', second_x.size())
        
        # Apply linear layer to continuous seconds feature
        #second_x = self.seconds_linear(second_x)
        second_x.unsqueeze_(-1)
        second_x = second_x.expand(second_x.size()[0], second_x.size()[1], self.d_model)
        #print("second_x shape = ", second_x.size())
        #print("second_x  = ", second_x)
        # Return combined tensor
        return hour_x + weekday_x + day_x + month_x + minute_x + second_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'ms': 6, 'us': 6, 'ns': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        #print('TimeFeatureEmbedding x = ', x.size())
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, seq_len, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        if embed_type != 'timeF':
            self.temporal_embedding = TemporalEmbeddingHighFreq(
                d_model=d_model, seq_len=seq_len, embed_type=embed_type,freq=freq
            ) if freq in ['ms','us','ns'] else TemporalEmbedding(
                d_model=d_model,embed_type=embed_type,freq=freq
            )
        else: 
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # value_embedding(x) and position_embedding(x) are embedding feature columns without datetime info, 
        # and temporal_embedding(x_mark) is the part embedding datetime info.
        # x = B x seq_len x Number of features / 
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        #print('DataEmbedding x = ', x.size())
        #print('DataEmbedding x_mark = ', x_mark.size())
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
