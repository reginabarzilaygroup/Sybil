import torch
import torch.nn as nn


class MultiAttentionPool(nn.Module):
    def __init__(self):
        super(MultiAttentionPool, self).__init__()
        params = {
            'num_chan': 512,
            'conv_pool_kernel_size': 11,
            'stride': 1
            }
        self.image_pool1 = Simple_AttentionPool_MultiImg(**params)
        self.volume_pool1 = Simple_AttentionPool(**params)

        self.image_pool2 = PerFrameMaxPool()
        self.volume_pool2 = Conv1d_AttnPool(**params)

        self.global_max_pool = GlobalMaxPool()

        self.multi_img_hidden_fc = nn.Linear(2 * 512, 512)
        self.hidden_fc = nn.Linear(3 * 512, 512)         

    def forward(self, x):
        #X dim: B, C, T, W, H
        output = {}
        
        image_pool_out1 = self.image_pool1(x) # contains keys: "multi_image_hidden", "image_attention"
        volume_pool_out1 = self.volume_pool1(image_pool_out1['multi_image_hidden'])  # contains keys: "hidden", "volume_attention"

        image_pool_out2 = self.image_pool2(x) # contains keys: "multi_image_hidden"
        volume_pool_out2 = self.volume_pool2(image_pool_out2['multi_image_hidden']) # contains keys: "hidden", "volume_attention"

        for pool_out, num in [(image_pool_out1, 1), (volume_pool_out1, 1), (image_pool_out2, 2), (volume_pool_out2, 2) ]:
            for key, val in pool_out.items():
                output['{}_{}'.format(key, num)] = val
    
        maxpool_out = self.global_max_pool(x)
        output['maxpool_hidden'] = maxpool_out['hidden']
        
        multi_image_hidden = torch.cat( [ image_pool_out1['multi_image_hidden'], image_pool_out2['multi_image_hidden']], dim = -2 )
        output['multi_image_hidden'] = self.multi_img_hidden_fc(multi_image_hidden.permute([0,2,1]).contiguous()).permute([0,2,1]).contiguous()

        hidden = torch.cat( [ volume_pool_out1['hidden'], volume_pool_out2['hidden'], output['maxpool_hidden']], dim = -1 )
        output['hidden'] = self.hidden_fc(hidden)

        return output 


class GlobalMaxPool(nn.Module):
    '''
    Pool to obtain the maximum value for each channel
    '''
    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict. output['hidden'] is (B, C)
        '''
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        hidden, _ = torch.max(x, dim=-1)
        return {'hidden': hidden}


class PerFrameMaxPool(nn.Module):
    '''
    Pool to obtain the maximum value for each slice in 3D input 
    '''
    def __init__(self):
        super(PerFrameMaxPool, self).__init__()
    
    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict. 
                + output['multi_image_hidden'] is (B, C, T)
        '''
        assert len(x.shape) == 5
        output = {}
        spatially_flat_size = (*x.size()[:3], -1)
        x = x.view(spatially_flat_size)
        output['multi_image_hidden'], _ = torch.max(x, dim=-1)
        return output


class Conv1d_AttnPool(nn.Module):
        '''
        Pool to learn an attention over the slices after convolution
        '''
        def __init__(self, **kwargs):
            super(Conv1d_AttnPool, self).__init__()
            self.conv1d = nn.Conv1d(kwargs['num_chan'], kwargs['num_chan'], kernel_size= kwargs['conv_pool_kernel_size'], stride=kwargs['stride'], padding= kwargs['conv_pool_kernel_size']//2, bias=False)
            self.aggregate = Simple_AttentionPool(**kwargs)
        
        def forward(self, x):
            '''
            args: 
                - x: tensor of shape (B, C, T)
            returns:
                - output: dict
                    + output['attention_scores']: tensor (B, C)
                    + output['hidden']: tensor (B, C)
            '''
            # X: B, C, N
            x = self.conv1d(x) # B, C, N'
            return self.aggregate(x)


class Simple_AttentionPool(nn.Module):
    '''
    Pool to learn an attention over the slices
    '''
    def __init__(self, **kwargs):
        super(Simple_AttentionPool, self).__init__()

        self.attention_fc = nn.Linear(kwargs['num_chan'], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, N)
        returns:
            - output: dict
                + output['volume_attention']: tensor (B, N)
                + output['hidden']: tensor (B, C)
        '''
        output = {}
        B = x.shape[0]
        spatially_flat_size = (*x.size()[:2], -1) #B, C, N

        x = x.view(spatially_flat_size)
        attention_scores = self.attention_fc(x.transpose(1,2)) #B, N, 1
        
        output['volume_attention'] = self.logsoftmax(attention_scores.transpose(1,2)).view(B, -1)
        attention_scores = self.softmax( attention_scores.transpose(1,2)) #B, 1, N
        
        x = x * attention_scores #B, C, N
        output['hidden'] = torch.sum(x, dim=-1)
        return output


class Simple_AttentionPool_MultiImg(nn.Module):
    '''
    Pool to learn an attention over the slices and the volume
    '''
    def __init__(self, **kwargs):
        super(Simple_AttentionPool_MultiImg, self).__init__()

        self.attention_fc = nn.Linear(kwargs['num_chan'], 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        '''
        args: 
            - x: tensor of shape (B, C, T, W, H)
        returns:
            - output: dict
                + output['image_attention']: tensor (B, T, W*H)
                + output['multi_image_hidden']: tensor (B, C, T)
                + output['hidden']: tensor (B, T*C)
        '''
        output = {} 
        B, C, T, W, H = x.size()
        x = x.permute([0,2,1,3,4])
        x = x.contiguous().view(B*T, C, W*H)
        attention_scores = self.attention_fc(x.transpose(1,2)) #BT, WH , 1
        
        output['image_attention'] = self.logsoftmax(attention_scores.transpose(1,2)).view(B, T, -1) 
        attention_scores = self.softmax( attention_scores.transpose(1,2)) #BT, 1, WH
        
        x = x * attention_scores #BT, C, WH
        x = torch.sum(x, dim=-1)
        output['multi_image_hidden'] = x.view(B, T, C).permute([0,2,1]).contiguous()
        output['hidden'] = x.view(B, T * C)
        return output
