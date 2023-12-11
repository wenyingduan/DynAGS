import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.AGCN import AVWGCN, AVWGCNT
import math
limit_a, limit_b, epsilon = -.1, 1.1, 1e-6


class _AGCRNCell_(nn.Module):
    def __init__(self, adj, node_num, hyper_model_dim, dim_in, dim_out, cheb_k, embed_dim, weight_decay=1., droprate_init=0.5, temperature=2./3.,
                 lamba=1., local_rep=False, sparse= True):
        super(_AGCRNCell_, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCNT(adj,  dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim, node_num)
        self.update = AVWGCNT(adj,  dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim, node_num)
        #l0
        self.prior_prec = weight_decay
        self.qz_loga =None
        self.qz_linear = nn.Linear(hyper_model_dim, node_num)
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.local_rep = local_rep
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
    def forward(self, x, state, node_embeddings, h):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        self.qz_loga = self.qz_linear(h)
        mask = self.sample_weights()
        z_r=self.gate(input_and_state, node_embeddings, mask)
        z_r = torch.sigmoid(z_r)
        #z_r, mask = torch.sigmoid(self.gate(input_and_state, node_embeddings, h))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        update = self.update(candidate, node_embeddings, mask)
        hc = torch.tanh(update)
        #hc = torch.tanh(self.update(candidate, node_embeddings, h)[0])
        h = r*state + (1-r)*hc
        return h, mask

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
        
    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self, node_embeddings):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        node_embeddings = node_embeddings
        supports = F.softmax(F.elu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        logpw_col = torch.sum(- (.5 * self.prior_prec * supports.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        #logpw = torch.sum((1 - self.cdf_qz(0)) )
        logpb = 0 
        return logpw + logpb
    
  

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.node_num
        expected_l0 = ppos * self.node_num
        
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps
            
    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.node_num,self.node_num)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        #return mask.view(self.in_features, 1) * self.weights
        return mask
        

class AGCRNCell(nn.Module):
    def __init__(self, adj, node_num, hyper_model_dim, dim_in, dim_out, cheb_k, embed_dim, weight_decay=1., droprate_init=0.5, temperature=2./3.,
                 lamba=1., local_rep=False, sparse= True):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(adj,  dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim, node_num)
        self.update = AVWGCN(adj,  dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim, node_num)
        #l0
        self.prior_prec = weight_decay
        self.qz_loga =None
        self.qz_linear = nn.Linear(hyper_model_dim, node_num)
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.local_rep = local_rep
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
    def forward(self, x, state, node_embeddings, h):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        self.qz_loga = self.qz_linear(h)
        mask = self.sample_weights()
        z_r=self.gate(input_and_state, node_embeddings, mask)
        z_r = torch.sigmoid(z_r)
        #z_r, mask = torch.sigmoid(self.gate(input_and_state, node_embeddings, h))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        update = self.update(candidate, node_embeddings, mask)
        hc = torch.tanh(update)
        #hc = torch.tanh(self.update(candidate, node_embeddings, h)[0])
        h = r*state + (1-r)*hc
        return h, mask

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
        
    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self, node_embeddings):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        node_embeddings = node_embeddings
        supports = F.softmax(F.elu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        logpw_col = torch.sum(- (.5 * self.prior_prec * supports.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        #logpw = torch.sum((1 - self.cdf_qz(0)) )
        logpb = 0 
        return logpw + logpb
    
  

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.node_num
        expected_l0 = ppos * self.node_num
        
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps
            
    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.node_num,self.node_num)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        #return mask.view(self.in_features, 1) * self.weights
        return mask
        
        
class AGLSTMCell(nn.Module):
    def __init__(self, node_num, hyper_model_dim, dim_in, dim_out, cheb_k, embed_dim, sparse):
        super(AGLSTMCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gcn= AVWGCN(hyper_model_dim, dim_in+self.hidden_dim, 4*dim_out, cheb_k, embed_dim, node_num, sparse)
     
    def forward(self, x, state, node_embeddings,h, mask = None):
       
        h,c = state
        batch_size = x.size(0)
       
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        h = h.to(x.device)
        c = c.to(x.device)
        
        input_and_state = torch.cat((x, h), dim=-1)
        
        ifgo, mask=self.gcn(input_and_state, node_embeddings,h, mask) 
        #o = self.gate(input_and_state, node_embeddings) 
       
        i,f,g,o = torch.chunk(ifgo,4,-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        new_c = f*c +i*g
        new_h = o*torch.tanh(new_c)
        return (new_h,new_c), mask

    def init_hidden_state(self, batch_size):
        state =torch.zeros(batch_size, 2, self.node_num, self.hidden_dim)
        
        return state
