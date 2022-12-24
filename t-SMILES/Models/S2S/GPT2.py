import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import math

from Models.Utils import ModelUtils

def GPT2_HParam():
    config = {
        'batch_size'            : 16,

        'optimizer'             : 'Adam',
        'optimizer_lr'          : 0.0005,
        'optimizer_weight_decay': 0.0001,

        'num_layers'            : 3,    #Number of hidden layers in the Transformer encoder
        'embedding_size'        : 512,  #Dimensionality of the embeddings and hidden states.
        'hidden_size'           : 512,

        'n_heads'               : 8,    #Number of attention heads for each attention layer in the Transformer encoder
        'n_ctx'                 : 1024, #Size of the causal mask (usually same as n_positions)
        'layer_norm_epsilon'    : 1e-05,
       
        'embd_pdrop'            : 0.1,
        'atten_dropout'         : 0.1,
        'resid_dropout'         : 0.1,

        'initializer_range'     : 0.02,

        'torchscript'           : False,
    }
    return config


class GPT2Base(nn.Module):
    def __init__(self, *inputs, **kwargs):
        super(GPT2Base, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std = self.hparam['initializer_range'])
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        return

    def tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.hparam['torchscript']:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
        return

class GPT2(GPT2Base):
    #The bare GPT2 Model transformer outputing raw hidden-states without any specific head on top.
    #it means there is no final output linear layer to target    
    def __init__(self, hparam, tparam):
        super(GPT2, self).__init__()
         
        self.hparam = hparam
        self.tparam = tparam
        self.device = tparam['device']

        self.output_hidden_states = True
        self.output_attentions = True
        hparam['output_attentions'] = self.output_attentions

        assert  hparam['embedding_size'] % hparam['n_heads'] == 0

        vocab_size = tparam['in_features']
        n_positions = tparam['in_channels']

        self.tok_embedding = nn.Embedding(vocab_size, hparam['embedding_size'])
        self.pos_embedding = nn.Embedding(n_positions, hparam['embedding_size'])
        self.dropout = nn.Dropout(hparam['embd_pdrop'])        
       
        self.blocks = nn.ModuleList([Block(hparam['n_ctx'], hparam, scale=True) for i in range(hparam['num_layers'])])

        self.layer_norm = BertLayerNorm(hparam['embedding_size'], eps = hparam['layer_norm_epsilon'])
         
        #self.linear_out = nn.Linear(hparam['embedding_size'], tparam['out_features'])

        self.apply(self.init_weights)
        #print(self)

        return

    def resize_token_embeddings(self, new_num_tokens):
        self.tok_embedding = self.get_resized_embeddings(self.tok_embedding, new_num_tokens)
        return self.tok_embedding

    def prune_heads(self, heads_to_prune):
        #heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        for layer, heads in heads_to_prune.items():
            self.blocks[layer].attn.prune_heads(heads)

    def forward(self, 
                input_ids,  #LongTensor(batch_size, sequence_length):int code of input samples
                position_ids = None,  #LongTensor(batch_size, sequence_length): [0, n_positions - 1] positions of each input sequence tokens in the position embeddings
                token_type_ids = None,#LongTensor(batch_size, sequence_length)  A parallel sequence of tokens (can be used to indicate various portions of the inputs).
                                      #The embeddings from these tokens will be summed with the respective token embeddings.
                                      #Indices are selected in the vocabulary (unlike BERT which has a specific vocabulary for segment indices).
                past = None,     #FloatTensor(one for each layer)that contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                                 #Can be used to speed up sequential decoding.
                head_mask = None, #FloatTensor(num_layers, num_heads) Mask to nullify selected heads of the self-attention modules.
                                 #Mask values selected in [0, 1]:
                                 #1:indicates the head is *not masked*; 0:indicates the head is *masked*.
                attention_mask = None, #FloatTensor(batch_size, sequence_length) Mask to avoid performing attention on padding token indices.
                                 #Mask values selected in [0, 1].
                                 #1: for tokens that are NOT MASKED ; 0: for MASKED tokens.
                ):
        if past is None:
            past_length = 0
            past = [None] * len(self.blocks)    #[None, None, None]
        else:
            past_length = past[0][0].size(-1)

        if position_ids is None:
            position_ids = torch.arange(past_length, 
                                        input_ids.size(-1) + past_length,
                                        dtype = torch.long, device = input_ids.device)  #torch.Size([120])  tensor[0...119]
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  #torch.Size([16, 120])
            
        # Prepare head mask if needed, 1.0 in head_mask menas we keep the head
        # attention_probs:(bsz x n_heads x N x N)
        # head_mask:(n_layer x batch x n_heads x N x N)
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.hparam['num_layers'], -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.hparam['num_layers']  #[None, None, None]

        input_shape = input_ids.size()      #torch.Size([16, 120])
        input_ids   = input_ids.view(-1, input_ids.size(-1)) #torch.Size([16, 120])
        position_ids = position_ids.view(-1, position_ids.size(-1))  #torch.Size([16, 120])

        inputs_embedding = self.tok_embedding(input_ids)      #torch.Size([16, 120, 512])  , sampling-first round torch.Size([1, 1, 512])
        position_embedding = self.pos_embedding(position_ids) #torch.Size([16, 120, 512])  , sampling-first round torch.Size([1, 1, 512])

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embedding = self.tok_embedding(token_type_ids)
        else:
            token_type_embedding = 0

        hidden_states = inputs_embedding + position_embedding + token_type_embedding  #torch.Size([16, 120, 512])
        hidden_states = self.dropout(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)  #torch.Size([16, 120, 512])

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.blocks, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states, layer_past, head_mask[i]) #[0:torch.Size([16, 120, 512]), 
                                                                     # 1:torch.Size([2, 16, 8, 120, 64]), 
                                                                     # 2:torch.Size([16, 8, 120, 120])]            
            hidden_states, present = outputs[:2]

            presents = presents + (present,)   #n * torch.Size([2, 16, 8, 120, 64])
            
            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.layer_norm(hidden_states)    #torch.Size([16, 120, 512])

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        if self.output_attentions:
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]  #torch.Size([16, -1, 120, 120])
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)   #shape[3* torch.Size([16, 8, 120, 120]) ]
            outputs = outputs + (all_attentions,)

        return outputs

class GPT2LanguageModel(GPT2Base):
    """
    The GPT2 Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings).
    """
    def __init__(self, hparam, tparam, ctoken):
        super(GPT2LanguageModel, self).__init__()
        self.hparam = hparam
        self.tparam = tparam
        self.device = tparam['device']

        self.ctoken      = ctoken        
        self.padding_idx = ctoken.pad_index

        self.transformer = GPT2(hparam, tparam)
        self.linear_out = nn.Linear(hparam['embedding_size'], tparam['out_features'])

        self.apply(self.init_weights)
        self.tie_weights()

        self.clip = tparam['clip']
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.padding_idx)
        self.optimizer, self.lr_step = ModelUtils.create_optimizer(self, self.hparam, self.tparam)
              
        self.to(self.device)

        print(self)
        return

    def tie_weights(self):
        #Make sure we are sharing the input and output embeddings.
        self.tie_or_clone_weights(self.linear_out, self.transformer.tok_embedding)
        return

    def forward(self, input_ids,
                position_ids=None,
                token_type_ids=None,
                labels=None,
                past=None,
                head_mask=None
                ):
        transformer_outputs = self.transformer(input_ids,   #torch.Size([16, 120])
                                               position_ids = position_ids,
                                               token_type_ids = token_type_ids,
                                               past = past,
                                               head_mask = head_mask
                                               )
        hidden_states = transformer_outputs[0]     #torch.Size([16, 120, 512])
        lm_logits = self.linear_out(hidden_states) #torch.Size([16, 120, 29])

        outputs = (lm_logits,) + transformer_outputs[1:]
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()

        outputs = self.forward(input_ids = inputs,
                             position_ids = None,
                             token_type_ids = None,
                             past = None,
                             head_mask = None,
                            )              # torch.Size([128, 120, 29])
        pred = outputs[0]
        output_dim = pred.shape[-1]  # 29

        output = pred.contiguous().view(-1, output_dim)     #torch.Size([119, 29])
        #trg = labels[:, 1:].contiguous().view(-1)          #torch.Size([119])
        trg = labels.contiguous().view(-1)                  #torch.Size([120])

        loss = self.criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)

        self.optimizer.step()

        return loss, pred

    def generate_seq_single(self, ctoken = None, max_len = 120, alg = 'argmax'):
        self.eval()

        input_ids = None
        position_ids = None
        token_type_ids = None
        past = None
        head_mask = None

        trg_indexes = [ctoken.start_index]   #trg_indexes = [1]

        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)  # first rounf:torch.Size([1, 1]),  last round:torch.Size([1, 120])
            #trg_mask = self.make_trg_mask(trg_tensor)  # last round: torch.Size([1, 1, 120, 120])

            with torch.no_grad():
                output = self.forward(input_ids = trg_tensor,
                                         position_ids = None,
                                         token_type_ids = None,
                                         past = None,
                                         head_mask = None,
                                         )
                if isinstance(output, (list, tuple, dict)):
                    attention = output[-1] #[0],[1],[3], torch.Size([1, 8, n, n]), 
                    output = output[0]          #torch.Size([1, 1, 46])

            #argmax and multinomial get the different tokens
            if alg =='argmax':   #just the raw scores, usually called logits
                #using argmax as a transition from the probability distribution to a token index produced.
                pred_token = output.argmax(2)[:, -1].item()
            else: #softmax & sampling which use distribution
                probs = torch.softmax(output, dim=-1)
                probs = probs[0][-1]
                pred_token = torch.multinomial(probs.view(-1), 1)[0].cpu().numpy()  #Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input
                pred_token = pred_token.min()                

            if pred_token == ctoken.end_index:
                break

            trg_indexes.append(pred_token)    #pred_char = ctoken.tokens[pred_token]

        #trg_tokens = ctoken.int_decode_single(trg_indexes)
        #return trg_tokens[1:], attention

        #return index for later operation
        return trg_indexes[1:], attention

    
    def generate_seq(self, nsamples = 100, max_len = 120, alg = 'softmax', show_atten = True):
        trg_samples = []
        for i in range(nsamples):
            trg, attention = self.generate_seq_single(self.ctoken, max_len = max_len, alg = alg)
            #added to handle index to string 
            trg = self.ctoken.int_decode_single(trg)


            #len(trg) = 59
            #attention[0,1,2].shape:torch.Size([1, 8, 60, 60])
            trg_samples.append(trg)
            
            if show_atten and len(trg) > 60 and len(trg) < 65:
                trg = list(trg)
                save_fname = 'attention_0'
                AttentionUtils.display_attention(trg, trg, attention[0], n_heads = self.hparam['n_heads'], save_fname = save_fname)
                save_fname = 'attention_1'
                AttentionUtils.display_attention(trg, trg, attention[1], n_heads = self.hparam['n_heads'], save_fname = save_fname)
                save_fname = 'attention_2'
                AttentionUtils.display_attention(trg, trg, attention[2], n_heads = self.hparam['n_heads'], save_fname = save_fname)
       
        return trg_samples

    def gernerate_seq_idx(self, nsamples = 100, max_len = 120, alg = 'softmax', show_atten = True):
        trg_samples = []
        for i in range(nsamples):
            trg, attention = self.generate_seq_single(self.ctoken, max_len = max_len, alg = alg)

            #len(trg) = 59
            #attention[0,1,2].shape:torch.Size([1, 8, 60, 60])
            trg_samples.append(trg)
            
            if show_atten and len(trg) < 10:
                trg = list(trg)
                AttentionUtils.display_attention(trg, trg, attention[0])
                AttentionUtils.display_attention(trg, trg, attention[1])
                AttentionUtils.display_attention(trg, trg, attention[2])
       
        return trg_samples

class Utils:
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def prune_linear_layer(layer, index, dim=0):
        """ Prune a linear layer (a model parameters) to keep only entries in index.
            Return the pruned layer as a new layer with requires_grad=True.
            Used to remove heads.
        """
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    def prune_conv1d_layer(layer, index, dim=1):
        """ Prune a Conv1D layer (a model parameters) to keep only entries in index.
            A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.
            Return the pruned layer as a new layer with requires_grad=True.
            Used to remove heads.
        """
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if dim == 0:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
        return new_layer

    def prune_layer(layer, index, dim=None):
        """ Prune a Conv1D or nn.Linear layer (a model parameters) to keep only entries in index.
            Return the pruned layer as a new layer with requires_grad=True.
            Used to remove heads.
        """
        if isinstance(layer, nn.Linear):
            return Utils.prune_linear_layer(layer, index, dim=0 if dim is None else dim)
        elif isinstance(layer, Conv1D):
            return Utils.prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
        else:
            raise ValueError("Can't prune layer of class {}".format(layer.__class__))
        return

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class MLP(nn.Module):
    def __init__(self,
                 n_state,  # in MLP: n_state=3072 (4 * embedding_size)
                 hparam):
        super(MLP, self).__init__()

        nx = hparam['embedding_size']
        self.c_fc= Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = Utils.gelu
        self.dropout = nn.Dropout(hparam['resid_dropout'])
        return

    def forward(self,x):
        h = self.c_fc(x)
        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, hparam, scale=False):
        super(Attention, self).__init__()
        #self.output_attentions =  hparam['output_attentions']
        self.output_attentions = True

        n_state = nx  # in Attention: nstate = nx = embedding_size

        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_heads = hparam['n_heads']
        self.split_size = n_state
        self.scale = scale

        self.c_atten = Conv1D(n_state * 3, nx)
        self.c_proj  = Conv1D(n_state, nx)

        self.atten_dropout = nn.Dropout(hparam['atten_dropout'])
        self.resid_dropout = nn.Dropout(hparam['resid_dropout'])
        return

    def prune_heads(self, heads):
        if len(heads) == 0:
            return

        mask = torch.ones(self.n_heads, self.split_size //self.n_heads)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous.eq(1)

        index = torch.arange(len(mask))[mask].long()
        index_atten = torch.cat([index,
                                 index + self.split_size,
                                 index + (2* self.split_size)
                                ])

        self.c_atten = Utils.prune_conv1d_layer(self.c_atten, index_atten, dim = 1)
        self.c_proj = Utils.prune_conv1d_layer(self.c_proj, index, dim = 1)

        self.split_size = (self.split_size //self.n_heads) * (self.n_heads - len(heads))
        self.n_heads = self.n_heads - len(heads)
        return

    def atten(self, q, k, v, head_mask = None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w/math.sqrt(v.size(-1))

        nd = w.size(-2)
        ns = w.size(-1)

        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w*b - 1e4*(1-b)

        w = nn.Softmax(dim = -1)(w)
        w = self.atten_dropout(w)

        #mask heads if needed
        if head_mask is not None:
            w = w* head_mask

        outputs = [torch.matmul(w,v)]
        if self.output_attentions:
            outputs.append(w)

        return outputs

    def merge_heads(self,x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k = False):
        new_x_shape = x.size()[:-1] + (self.n_heads, x.size(-1) // self.n_heads)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past = None, head_mask = None):
        #input x: torch.Size([16, 120, 512]), 
        x = self.c_atten(x) #torch.Size([16, 120, 1536]), self.split_size = 512

        query, key, value  = torch.chunk(input = x, chunks = 3, dim=2)
        ##query, key, value = x.split(split_size_or_sections = self.split_size, dim = 2)
        #query, key, value have the same size: torch.Size([16, 120, 512])

        query = self.split_heads(query)         #torch.Size([16, 8, 120, 64])
        key = self.split_heads(key, k = True)   #torch.Size([16, 8, 64, 120])
        value = self.split_heads(value)         #torch.Size([16, 8, 120, 64])

        if layer_past is not None:
            past_key    = layer_past[0].transpose(-2,-1)
            past_value  = layer_past[1]
            key = torch.cat((past_key, key), dim = -1)
            vlaue = torch.cat((past_value, value), dim = -2)

        present = torch.stack((key.transpose(-2,-1), value))       #torch.Size([2, 16, 8, 120, 64])

        atten_outputs = self.atten(query, key, value, head_mask)   #atten_outputs[0]: torch.Size([16, 8, 120, 64])
        a = atten_outputs[0]        #torch.Size([16, 8, 120, 64])

        a = self.merge_heads(a)     #torch.Size([16, 120, 512])
        a = self.c_proj(a)          #torch.Size([16, 120, 512])
        a = self.resid_dropout(a)   ##torch.Size([16, 120, 512])

        outputs = [a, present ] + atten_outputs[1:]
        return outputs   #[0:torch.Size([16, 120, 512]), 1:torch.Size([2, 16, 8, 120, 64])]

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Block(nn.Module):
    def __init__(self, n_ctx, hparam, scale = False):
        super(Block, self).__init__()

        nx = hparam['embedding_size']
        self.ln_1 = BertLayerNorm(hidden_size = nx, eps = hparam['layer_norm_epsilon'])

        self.atten = Attention(nx = nx, 
                               n_ctx = n_ctx, 
                               hparam = hparam, 
                               scale = scale)

        self.ln_2 = BertLayerNorm(hidden_size = nx, eps = hparam['layer_norm_epsilon'])

        self.mlp = MLP(4*nx, hparam)

        return

    def forward(self, x, layer_past = None, head_mask = None):
        #input x: torch.Size([16, 120, 512])
        h = self.ln_1(x)    #torch.Size([16, 120, 512])
        output_atten = self.atten(h, layer_past = layer_past, head_mask = head_mask)   #[0:torch.Size([16, 120, 512]), 1:torch.Size([2, 16, 8, 120, 64])]
        a = output_atten[0]     #torch.Size([16, 120, 512])

        x = x + a           #torch.Size([16, 120, 512])
        h = self.ln_2(x)    #torch.Size([16, 120, 512])
        m = self.mlp(h)     #torch.Size([16, 120, 512])
        x = x + m           #torch.Size([16, 120, 512])

        outputs = [x] + output_atten[1:]  #[0:torch.Size([16, 120, 512]), 1:torch.Size([2, 16, 8, 120, 64])]
        return outputs
