import torch
import torch.nn as nn
from model.AGCRNCell import AGCRNCell, _AGCRNCell_, AGLSTMCell
from HYPER import HyperNet
class AVWDCRNN(nn.Module):
    def __init__(self, adj,  node_num, hyper_model_dim, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in*2
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(adj, node_num, hyper_model_dim, self.input_dim, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(_AGCRNCell_(adj, node_num, hyper_model_dim, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, h):
        
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        mask_list = []
       
        for i in range(self.num_layers):
            state = init_state[i]
            #state = (state[:,0],state[:,1])
            inner_states = []
            mask_list_up =[]
            for t in range(seq_length):
                
                state, mask= self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, h[:, :, t, :]) 
                inner_states.append(state)
                mask_list.append(mask)
              
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        #print(torch.stack(mask_list).size())
        return current_inputs, output_hidden, torch.stack(mask_list)
   
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args, adj):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.window = args.lag
        self.scale = args.scale
        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
    
        
        self.hypernet = HyperNet(int(args.hyper_horizon/12), args.rnn_units, args.embed_dim, args.num_nodes)
        self.main_weights_pool = nn.Parameter(torch.FloatTensor(args.embed_dim, args.rnn_units, args.input_dim))
        self.encoder = AVWDCRNN(adj, args.num_nodes, args.hyper_model_dim, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)
       
        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.layernorm = nn.LayerNorm(self.hidden_dim, eps=1e-12)
        self.output_dropout = nn.Dropout(0.1)
    def forward(self, hyper_source, source, targets,  teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        #hyper_outputs = self.Model(hyper_source) #hyper_outputs: B,L1, N,D
        init_state = self.encoder.init_hidden(source.shape[0])
        h = self.hypernet(hyper_source, source)
    
        weights_h = torch.einsum('nd,dhi->nhi', self.node_embeddings, self.main_weights_pool) #N,hidden,I
        
        #source = torch.einsum('bnld, ndi->blni',h,weights_h)+source
        source = torch.cat([torch.einsum('bnld, ndi->blni',h,weights_h), source],-1)
        #print(source.size())
        output, _ , sp  = self.encoder(source, init_state, self.node_embeddings, h)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        output = self.output_dropout(self.layernorm(output))
        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output, sp

    def regularization_cell_1(self):
       
        
        #reg = self.encoder.dcrnn_cells[0].gate._reg_w(self.node_embeddings)+self.encoder.dcrnn_cells[1].gate._reg_w(self.node_embeddings)
        #Reg = self.encoder.dcrnn_cells[0].gcn._reg_w(self.node_embeddings)+self.encoder.dcrnn_cells[1].gcn._reg_w(self.node_embeddings)
        Reg = 0
        for i in range(len(self.encoder.dcrnn_cells)):
            Reg = Reg+self.encoder.dcrnn_cells[i]._reg_w(self.node_embeddings)
            #+self.encoder.dcrnn_cells[i].update._reg_w(self.node_embeddings)
        
            
        return - ((1. / self.scale)*(Reg))
        
    
