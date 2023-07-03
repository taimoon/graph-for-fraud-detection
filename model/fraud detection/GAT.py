import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import pickle

class myGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads = 1, 
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(myGAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels # node features input dimension
        self.out_channels = out_channels # node level output dimension
        self.heads = heads # No. of attention heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

        # Initialization
        self.lin_l = nn.Linear(in_channels, heads*out_channels)
        self.lin_r = self.lin_l
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels).float())
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels).float())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels # DIM：H, outC

        # Linearly transform node feature matrix.
        x_source = self.lin_l(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]
        x_target = self.lin_r(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]

        # Alphas will be used to calculate attention later
        alpha_l = (x_source * self.att_l).sum(dim=-1) # DIM: [nodes, H, outC] x [H, outC] => [nodes, H]
        alpha_r = (x_target * self.att_r).sum(dim=-1) # DIM: [nodes, H, outC] x [H, outC] => [nodes, H]

        #  Start propagating messages (runs message and aggregate)
        out = self.propagate(edge_index, x=(x_source, x_target), alpha=(alpha_l, alpha_r),size=size) # DIM: [nodes, H, outC]
        out = out.view(-1, self.heads * self.out_channels) # DIM: [nodes, H * outC]

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        # Calculate attention for edge pairs
        attention = F.leaky_relu((alpha_j + alpha_i), self.negative_slope) # EQ(1) DIM: [Edges, H]
        attention = softmax(attention, index, ptr, size_i) # EQ(2) DIM: [Edges, H] | This softmax only calculates it over all neighbourhood nodes
        attention = F.dropout(attention, p=self.dropout, training=self.training) # DIM: [Edges, H]

        # Multiple attention with node features for all edges
        out = x_j * attention.unsqueeze(-1)  # EQ(3.1) [Edges, H, outC] x [Edges, H] = [Edges, H, outC];

        return out

    def aggregate(self, inputs, index, dim_size = None):
        # EQ(3.2) For each node, aggregate messages for all neighbourhood nodes 
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, 
                                    dim_size=dim_size, reduce='sum') # inputs (from message) DIM: [Edges, H, outC] => DIM: [Nodes, H, outC]
        return out


class myGATv2(MessagePassing):
    def __init__(self, in_channels, out_channels, heads = 1,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(myGATv2, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None
        self._alpha = None
        # self.lin_l is the linear transformation that you apply to embeddings 
        # BEFORE message passing.
        self.lin_l =  nn.Linear(in_channels, heads*out_channels)
        self.lin_r = self.lin_l

        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.reset_parameters()

    #initialize parameters with xavier uniform
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels # DIM：H, outC
        #Linearly transform node feature matrix.
        x_source = self.lin_l(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]
        x_target = self.lin_r(x).view(-1,H,C) # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC]
        
        #  Start propagating messages (runs message and aggregate)
        out= self.propagate(edge_index, x=(x_source,x_target),size=size) # DIM: [nodes, H, outC]
        out= out.view(-1, self.heads * self.out_channels)       # DIM: [nodes, H * outC]
        alpha = self._alpha
        self._alpha = None
        return out

    #Process a message passing
    def message(self, x_j,x_i,  index, ptr, size_i):
        #computation using previous equationss
        x = x_i + x_j                               
        x  = F.leaky_relu(x, self.negative_slope)   # See Equation above: Apply the non-linearty function
        alpha = (x * self.att).sum(dim=-1)          # Apply attnention "a" layer after the non-linearity 
        alpha = softmax(alpha, index, ptr, size_i)  # This softmax only calculates it over all neighbourhood nodes
        self._alpha = alpha
        alpha= F.dropout(alpha,p=self.dropout,training=self.training)
        # Multiple attention with node features for all edges
        out= x_j*alpha.unsqueeze(-1)  

        return out
    #Aggregation of messages
    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, 
                                    dim_size=dim_size, reduce='sum')  
        return out

class GATmodif(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,args):
        self.args = args
        super(GATmodif, self).__init__()
        #use our gat message passing 
        self.conv1 = myGAT(input_dim, hidden_dim,heads=args['heads'])
        self.conv2 = myGAT(args['heads']  *hidden_dim, hidden_dim,heads=args['heads']) 

        self.post_mp = nn.Sequential(
            nn.Linear(args['heads'] * hidden_dim, hidden_dim), nn.Dropout(args['dropout'] ), 
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self, data, adj=None):
        args = self.args
        x, edge_index = data.x, data.edge_index
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)

        # MLP output
        x = self.post_mp(x)
        return F.sigmoid(x)

class GATv2modif(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,args):
        self.args = args
        super(GATv2modif, self).__init__()
        #use our gat message passing 
        self.conv1 = myGATv2(input_dim, hidden_dim,heads=args['heads']) 
        self.conv2 = myGATv2(args['heads'] *hidden_dim, hidden_dim,heads=args['heads'])

        self.post_mp = nn.Sequential(
            nn.Linear(args['heads']  * hidden_dim, hidden_dim), nn.Dropout(args['dropout'] ), 
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self, data, adj=None):
        args = self.args
        x, edge_index = data.x, data.edge_index
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)

        # MLP output
        x = self.post_mp(x)
        return F.sigmoid(x)

class GATmodif_3layer(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, args):
        self.args = args
        super(GATmodif_3layer, self).__init__()
        #use our gat message passing 
        self.conv1 = myGAT(input_dim, hidden_dim,heads=args['heads'])
        self.conv2 = myGAT(args['heads'] * hidden_dim, hidden_dim,heads=args['heads']) 
        self.conv3 = myGAT(args['heads'] * hidden_dim, hidden_dim,heads=args['heads']) 
        
        self.post_mp = nn.Sequential(
            nn.Linear(args['heads'] * hidden_dim, hidden_dim), nn.Dropout(args['dropout'] ), 
            nn.Linear(hidden_dim, output_dim))
        
    def forward(self, data, adj=None):
        args = self.args
        x, edge_index = data.x, data.edge_index
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)

        # MLP output
        x = self.post_mp(x)
        return F.sigmoid(x)