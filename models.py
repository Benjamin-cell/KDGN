import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dnc import DNC
from layers import GraphConvolution

import math
from torch.nn.parameter import Parameter

#### 2022-3-27 ********
class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh()
        )

        self.gate_layer = nn.Linear(128, 64)

        self.tran = nn.Linear(128,64)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates + seq_masks
        p_attn = F.softmax(gates, dim=-1)

        # p_attn = p_attn.unsqueeze(-1)

        seqs = self.tran(seqs)


        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):

        q = self.weight

        weight = torch.mul(self.weight, mask) ## 矩阵对应位相乘，形状必须一致
        output = torch.mm(input, weight)  ## 矩阵相乘，有相同维度即可乘，维度可变

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(in_dim))
        self.fc = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):

        weights = torch.matmul(x.view(-1, x.shape[1]), self.cls_vec)
        weights = self.softmax(weights.view(x.shape[0], -1))
        x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), weights.unsqueeze(-1)).squeeze()
        x = x + self.cls_vec
        x = self.fc(x)
        x = x + self.cls_vec
        return x


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden



    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)


        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs




        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)

        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors


class MedTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5) -> None:
        super(MedTransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2p_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.nhead = nhead

        # self.align = nn.Linear(d_model, d_model)

    def forward(self, input_medication_embedding, input_medication_memory, input_disease_embdding, input_proc_embedding,
                input_medication_self_mask, d_mask, p_mask):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            input_medication_embedding: [*, max_med_num+1, embedding_size]
        Shape:
            see the docs in Transformer class.
        """
        input_len = input_medication_embedding.size(0)
        tgt_len = input_medication_embedding.size(1)

        # [batch_size*visit_num, max_med_num+1, max_med_num+1]
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len, input_len * self.nhead,
                                                               input_disease_embdding.device)
        self_attn_mask = subsequent_mask + input_medication_self_mask

        x = input_medication_embedding + input_medication_memory

        x = self.norm1(x + self._sa_block(x, self_attn_mask))
        # attentioned_disease_embedding = self._m2d_mha_block(x, input_disease_embdding, d_mask)
        # attentioned_proc_embedding = self._m2p_mha_block(x, input_proc_embedding, p_mask)
        # x = self.norm3(x + self._ff_block(torch.cat([attentioned_disease_embedding, self.align(attentioned_proc_embedding)], dim=-1)))
        x = self.norm2(
            x + self._m2d_mha_block(x, input_disease_embdding, d_mask) + self._m2p_mha_block(x, input_proc_embedding,
                                                                                             p_mask))
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _m2d_mha_block(self, x, mem, attn_mask):
        x = self.m2d_multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    need_weights=False)[0]
        return self.dropout2(x)

    def _m2p_mha_block(self, x, mem, attn_mask):
        x = self.m2p_multihead_attn(x, mem, mem,
                                    attn_mask=attn_mask,
                                    need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def generate_square_subsequent_mask(self, sz: int, batch_size: int, device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        return mask


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

'''

Our model

'''

import torch
from graph_transformer_pytorch import GraphTransformer


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        self.w = adj.shape[0]
        adj = self.normalize(adj + np.eye(256))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(256).to(device)

        self.gcn1 = GraphConvolution(256, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, 64) ## graph_transformer 改为16

    def forward(self):


        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        edge = self.adj

        return node_embedding ,edge

    def normalize(self, mx):
        """Row-normalize sparse matrix"""


        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class KDGN(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj,   ddi_mask_H, MPNNSet, N_fingerprints, average_projection,  emb_dim=64,device=torch.device('cuda'),ddi_in_memory=True):
        super(KDGN, self).__init__()
        K = len(vocab_size)
        # self.prescription_net = nn.Sequential(
        #     nn.Linear(112, 112 * 4),
        #     nn.ReLU(),
        #     nn.Linear(112 * 4, 112)
        # )
        self.emb_dim = 64
        self.nhead = 2
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.63)

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim*2, batch_first=True) for _ in range(K-1)])

        self.e = nn.ModuleList([nn.GRU(64, 64, batch_first=True) for _ in range(K - 1)])
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
############
        self.MED_PAD_TOKEN = vocab_size[2] + 2
        self.med_embedding = nn.Sequential(
            # 添加padding_idx，表示取0向量
            nn.Embedding(vocab_size[2] + 3, emb_dim, self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )
################


        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))


        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

        self.z = nn.Linear(256,112)

        self.q = nn.Linear(16384,256)
        self.bb = nn.Linear(128,153)

        self.tra = nn.Linear(128,153)
        self.tran = nn.Linear(4096,153)
        self.transpor = nn.Linear(128,1)
## 2022-3-27##########################################
        # 聚合单个visit内的diag和proc得到visit-level的表达
        self.diag_self_attend = SelfAttend(32)
        self.proc_self_attend = SelfAttend(32)



        self.medication_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)

        self.decoder = MedTransformerDecoder(emb_dim, self.nhead, dim_feedforward=emb_dim * 2, dropout=0.2,
                                             layer_norm_eps=1e-5) ## 未用到

        # bipartite local embedding 二分局部嵌入
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)  ## 或许是将ddi_mask矩阵中的药物屏蔽掉

        # MPNN global embedding
        self.MPNN_molecule_Set = list(zip(*MPNNSet))

        self.MPNN_emb = MolecularGraphNeuralNetwork(N_fingerprints, emb_dim, layer_hidden=2, device=device).forward(
            self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(average_projection.to(device=self.device), self.MPNN_emb.to(device=self.device))
        self.MPNN_emb.to(device=self.device)
        # self.MPNN_emb = torch.tensor(self.MPNN_emb, requires_grad=True)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])



        # self.attavg = AttentionPooling(112)



        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)


        self.abc = nn.Linear(153,112)

        self.ppp = nn.Linear(64,1)

        # self.residual = ResidualBlock(1, 64, 5, 1, True, 0.1)

### 2022-3-27 打分机制
    def calc_cross_visit_scores(self, visit_diag_embedding, visit_proc_embedding):
        """
        visit_diag_embedding: (batch * visit_num * emb)
        visit_proc_embedding: (batch * visit_num * emb)
        """
        max_visit_num = visit_diag_embedding.size(1)
        batch_size = visit_diag_embedding.size(0)

        new = visit_diag_embedding.size(2)

        # mask表示每个visit只能看到自己之前的visit
        mask = (torch.triu(torch.ones((max_visit_num, max_visit_num), device=self.device)) == 1).transpose(0,
                                                                                                           1)  # 返回一个下三角矩阵
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)  # batch * max_visit_num * max_visit_num

        # 每个visit后移一位
        padding = torch.zeros((batch_size, 1, new), device=self.device).float()


        diag_keys = torch.cat([padding, visit_diag_embedding[:, :-1, :]], dim=1)  # batch * max_visit_num * emb
        proc_keys = torch.cat([padding, visit_proc_embedding[:, :-1, :]], dim=1)

        # 得到每个visit跟自己前面所有visit的score
        diag_scores = torch.matmul(visit_diag_embedding, diag_keys.transpose(-2, -1)) \
                      / math.sqrt(visit_diag_embedding.size(-1))
        proc_scores = torch.matmul(visit_proc_embedding, proc_keys.transpose(-2, -1)) \
                      / math.sqrt(visit_proc_embedding.size(-1))
        # 1st visit's scores is not zero!
        scores = F.softmax(diag_scores + proc_scores + mask, dim=-1)

        ###### case study
        # 将第0个val置0，然后重新归一化
        # scores_buf = scores
        # scores_buf[:, :, 0] = 0.
        # scores_buf = scores_buf / torch.sum(scores_buf, dim=2, keepdim=True)

        # print(scores_buf)
        return scores





    def forward(self, input):
        # input (adm, 3, codes)
        device = self.device
        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        i3_seq = []
        max_med_num = 16 ##会变

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:

            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i3 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device))))


            ##  此处adm[0]代表diagnosis文件，adm[1]代表procedure文件；
            i1_seq.append(i1)
            i2_seq.append(i2)
            i3_seq.append(i3)
        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)
        i3_seq = torch.cat(i3_seq, dim=1)

        o1, h1 = self.encoders[0](
            i1_seq
        ) # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        o2, h2 = self.encoders[1](
            i2_seq
        )
        o3, h3 = self.e[1](
            i3_seq
        )



        max_visit_num = 1
        # print(o1.shape,h1.shape,'this is the o1')


        # print(o2,h2,'this is the o2')
## 2022-3-27 此处要对诊疗和程序部分信息分别做注意力，然后使用分层选择机制打分；

        # 1.1 encode visit-level diag and proc representations
        # visit_diag_embedding = self.diag_self_attend(o1)
        # visit_proc_embedding = self.proc_self_attend(o2)

        # print(visit_diag_embedding.shape) 1*64*128



        visit_diag_embedding = o1.view(64, max_visit_num, -1)

        visit_proc_embedding = o2.view(64, max_visit_num, -1)

        #
        # dec_hidden = self.decoder(input_medication_embedding=input_medication_embs,
        #                           input_medication_memory=input_medication_memory,
        #                           input_disease_embdding=input_disease_embedding.view(batch_size * max_visit_num,
        #                                                                               max_diag_num, -1),
        #                           input_proc_embedding=input_proc_embedding.view(batch_size * max_visit_num,
        #                                                                          max_proc_num, -1),
        #                           input_medication_self_mask=medication_self_mask,
        #                           d_mask=m2d_mask_matrix,
        #                           p_mask=m2p_mask_matrix)





        # 1.3 计算 visit-level的attention score
        # [batch_size, max_visit_num, max_visit_num]
        cross_visit_scores = self.calc_cross_visit_scores(visit_diag_embedding, visit_proc_embedding)

        batch_size=32

        # 1.4计算上一个药物表示
        # 3. 构造一个last_seq_medication，表示上一次visit的medication，第一次的由于没有上一次medication，用0填补（用啥填补都行，反正不会用到）
        # last_seq_medication = torch.full((batch_size, 1, 1), 0).to(device)
        # print(last_seq_medication.shape)



        last_seq_medication = h3
        # last_seq_medication = torch.cat([last_seq_medication, i3_seq], dim=2).to(device)
        # m_mask_matrix矩阵同样也需要后移


        # last_m_mask = torch.full((batch_size, 1, max_med_num), -1e9)  # 这里用较大负值，避免softmax之后分走了概率
        # last_m_mask = torch.cat([last_m_mask, m_mask_matrix[:, :,:-1 ]], dim=1)
        # 对last_seq_medication进行编码
        last_seq_medication = last_seq_medication.long()

        # print(last_seq_medication.shape,'*&^%$#')
        last_seq_medication_emb = self.med_embedding(last_seq_medication)
        # print(last_seq_medication_emb.shape, '*&^%$#')

        # last_m_enc_mask = last_m_mask.view(batch_size * max_visit_num, max_med_num).unsqueeze(dim=1).unsqueeze(
        #     dim=1).repeat(1, self.nhead, max_med_num, 1)
        # last_m_enc_mask = last_m_enc_mask.view(batch_size * max_visit_num * self.nhead, max_med_num, max_med_num)
        last_seq_medication_emb = torch.squeeze(last_seq_medication_emb)


        encoded_medication = self.medication_encoder(last_seq_medication_emb,src_mask=None)  # (batch*seq, max_med_num, emb_dim)





        prob_g = F.softmax(cross_visit_scores, dim=-1)

        prob_c_to_g = torch.zeros_like(prob_g).to(self.device) # (batch, max_visit_num * input_med_num, voc_size[2]+2)

        copy_source = last_seq_medication.view(batch_size, 1, -1).repeat(1, max_visit_num * max_med_num, 1)


        # prob_c_to_g.scatter_add_(2, copy_source, cross_visit_scores)
        # prob_c_to_g = prob_c_to_g.view(batch_size, max_visit_num, max_med_num, -1)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0) # (seq, dim*4)
        queries = self.query(patient_representations) # (seq, dim)

        # queries = self.residual(queries)

        # graph memory module

        query = queries[-1:] # (1,dim)


        drug_memory,edge = self.ehr_gcn()
        drug_memory = drug_memory.t()
        drug_memory = self.z(drug_memory)
        drug_memory = drug_memory.t()


        # if self.ddi_in_memory:
        #     drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        # else:


        # c = torch.stack([drug_memory, drug_memory], 0)
        # d = c.view(1,32,-1)
        # e = edge.view(4,-1)
        # f = self.q(e)
        # #
        # # # print(drug_memory.shape, c.shape,d.shape,e.shape, f.shape,'************')
        # model = GraphTransformer(
        #     dim=256,
        #     depth=6,
        #
        #     # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
        #     with_feedforwards=True,
        #     # whether to add a feedforward after each attention layer, suggested by literature to be needed
        #     gated_residual=True,  # to use the gated residual to prevent over-smoothing
        #     rel_pos_emb=True  # set to True if the nodes are ordered, default to False
        #
        # )
        # mask = torch.ones(1, 32).bool().to(device=device)
        # model.to(device=device)
        # nodes, edges = model(d, f, mask=mask)
        # # #
        # # #
        # #
        # node = nodes.squeeze()
        # # print(node.shape,) ## 153,64
        # node = node.view(64,-1)
        # drug_memory = self.bb(node)
        # drug_memory = drug_memory.t()



        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)] # (seq-1, dim)

            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)


        '''O:read from global memory bank and dynamic memory bank'''



        fun2 = torch.mm(query, drug_memory.t())


        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)
        w1 = F.sigmoid(fun2)
        w2 = 1 - w1



        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            # print(weighted_values,'*****************************************')
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
            # print(fact2.shape)
            # print(fact2,'')
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        # drug = self.vocab_size[2]
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        # generate_prob = F.sigmoid(self.W_z(dec_hidden))



        # MPNN embedding

        MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))




        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))

        # local embedding
        bipartite_emb = self.bipartite_output(F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t())

        result = torch.mul(bipartite_emb, MPNN_att)



        prob_g = prob_g.squeeze(dim=1)
        b = prob_g * encoded_medication
        b = b.view(1,-1)
        b = self.tran(b)



        b = self.abc(b)


        a = w1*( 0.85*output+ result )+ 1.85*b
        # a = output , drug


        if self.training:

            # neg_pred_prob = F.sigmoid(output)
            # neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            # batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()


            return a
        else:

            return a

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)


'''
DMNC
'''
class DMNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(DMNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=0.3)

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0,
            independent_linears=False
        ) for _ in range(K - 1)])

        self.decoder = nn.GRU(emb_dim + emb_dim * 2, emb_dim * 2,
                              batch_first=True)  # input: (y, r1, r2,) hidden: (hidden1, hidden2)
        self.interface_weighting = nn.Linear(emb_dim * 2, 2 * (emb_dim + 1 + 3))  # 2 read head (key, str, mode)
        self.decoder_r2o = nn.Linear(2 * emb_dim, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 2, vocab_size[2] + 2)

    def forward(self, input, i1_state=None, i2_state=None, h_n=None, max_len=20):
        # input (3, code)
        i1_input_tensor = self.embeddings[0](
            torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
        i2_input_tensor = self.embeddings[1](
            torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        o1, (ch1, m1, r1) = \
            self.encoders[0](i1_input_tensor, (None, None, None) if i1_state is None else i1_state)
        o2, (ch2, m2, r2) = \
            self.encoders[1](i2_input_tensor, (None, None, None) if i2_state is None else i2_state)

        # save memory state
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)

        predict_sequence = [self.token_start] + input[2]
        if h_n is None:
            h_n = torch.cat([ch1[0], ch2[0]], dim=-1)

        output_logits = []
        r1 = r1.unsqueeze(dim=0)
        r2 = r2.unsqueeze(dim=0)

        if self.training:
            for item in predict_sequence:
                # teacher force predict drug
                item_tensor = self.embeddings[2](
                    torch.LongTensor([item]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output_logits.append(output)
        else:
            item_tensor = self.embeddings[2](
                torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
            for idx in range(max_len):
                # predict
                # teacher force predict drug
                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(dim=1),
                                              read_strengths[:, 0].unsqueeze(dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(dim=1),
                                              read_strengths[:, 1].unsqueeze(dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)

                input_token = torch.argmax(output, dim=-1)
                input_token = input_token.item()
                item_tensor = self.embeddings[2](
                    torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        return torch.cat(output_logits, dim=0), i1_state, i2_state, h_n

    def read_from_memory(self, dnc, read_key, read_str, read_mode, m_hidden):
        read_vectors, hidden = dnc.memories[0].read(read_key, read_str, read_mode, m_hidden)
        return read_vectors, hidden

    def decode_read_variable(self, input):
        w = 64
        r = 2
        b = input.size(0)

        input = self.interface_weighting(input)
        # r read keys (b * w * r)
        read_keys = F.tanh(input[:, :r * w].contiguous().view(b, r, w))
        # r read strengths (b * r)
        read_strengths = F.softplus(input[:, r * w:r * w + r].contiguous().view(b, r))
        # read modes (b * 3*r)
        read_modes = F.softmax(input[:, (r * w + r):].contiguous().view(b, r, 3), -1)
        return read_keys, read_strengths, read_modes


'''
Leap
'''
class Leap(nn.Module):
    def __init__(self, voc_size, emb_dim=128, device=torch.device('cpu:0')):
        super(Leap, self).__init__()
        self.voc_size = voc_size
        self.device = device
        self.SOS_TOKEN = voc_size[2]
        self.END_TOKEN = voc_size[2]+1

        self.enc_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], emb_dim, ),
            nn.Dropout(0.3)
        )
        self.dec_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 2, emb_dim, ),
            nn.Dropout(0.3)
        )

        self.dec_gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim*2, 1)

        self.output = nn.Linear(emb_dim, voc_size[2]+2)


    def forward(self, input, max_len=20):
        device = self.device
        # input (3, codes)
        input_tensor = torch.LongTensor(input[0]).to(device)
        # (len, dim)
        input_embedding = self.enc_embedding(input_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        output_logits = []
        hidden_state = None
        if self.training:
            for med_code in [self.SOS_TOKEN] + input[2]:
                dec_input = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)

                if hidden_state is None:
                    hidden_state = dec_input

                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1) # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1) # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1) # (1, len)
                input_embedding = attn_weight.mm(input_embedding) # (1, dim)

                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0) # (1,dim)

                output_logits.append(self.output(F.relu(hidden_state)))

            return torch.cat(output_logits, dim=0)

        else:
            for di in range(max_len):
                if di == 0:
                    dec_input = torch.LongTensor([[self.SOS_TOKEN]]).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0) # (1,dim)
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(input_embedding.size(0), 1)  # (len, dim)
                combined_input = torch.cat([hidden_state_repeat, input_embedding], dim=-1)  # (len, dim*2)
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)  # (1, len)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)
                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
                                               hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)
                output = self.output(F.relu(hidden_state))
                topv, topi = output.data.topk(1)
                output_logits.append(F.softmax(output, dim=-1))
                dec_input = topi.detach()
            return torch.cat(output_logits, dim=0)

'''
Retain
'''
class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.3)
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, input):
        device = self.device
        # input: (visit, 3, codes )
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in input])
        input_np = []
        for visit in input:
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0]))
            input_tmp.extend(list(np.array(visit[2]) + self.voc_size[0] + self.voc_size[1]))
            if len(input_tmp) < max_len:
                input_tmp.extend( [self.input_len]*(max_len - len(input_tmp)) )

            input_np.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(input_np).to(device)) # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1) # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0)) # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0)) # h: (1, visit, emb)

        g = g.squeeze(dim=0) # (visit, emb)
        h = h.squeeze(dim=0) # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1) # (visit, 1)
        attn_h = F.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        return self.output(c)

'''
RF in train_LR.py
'''


