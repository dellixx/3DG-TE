from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries, filters, year2id = {},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestamp)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    if queries.shape[1]>4: #time intervals exist
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start=-int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#','0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end =-int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#','0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
                                if start>=key[0] and start<=key[1]:
                                    start_idx = time_idx
                                if end>=key[0] and end<=key[1]:
                                    end_idx = time_idx


                            if start_idx < 0:
                                start_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append([int(triple[0]), int(triple[1])+self.sizes[1]//4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()

                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries) + self.score(end_queries)
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size] # 500, 4
                        q = self.get_queries(these_queries) # 500, 400
                        """
                        if use_left_queries:
                            lhs_queries = torch.ones(these_queries.size()).long().cuda()
                            lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
                            lhs_queries[:,0] = these_queries[:,2]
                            lhs_queries[:,2] = these_queries[:,0]
                            lhs_queries[:,3] = these_queries[:,3]
                            q_lhs = self.get_lhs_queries(lhs_queries)

                            scores = q @ rhs +  q_lhs @ rhs
                            targets = self.score(these_queries) + self.score(lhs_queries)
                        """
                        
                        scores = q @ rhs 
                        targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        if queries.shape[1]>4:
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]                            
                        else:    
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class GaussTKGE_3D(TKBCModel):

    def __init__(self, sizes: Tuple[int, int, int, int], rank: int,no_time_emb=False, init_size: float = 1e-2):
        super(GaussTKGE_3D, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.mat_n = int(np.sqrt(rank*2))
        self.rank_ = rank * 9
        self.half_rank_ = int(self.rank_ / 2)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0],  self.rank_, sparse=True),
            nn.Embedding(sizes[1],  self.rank_, sparse=True),
            nn.Embedding(sizes[3],  self.rank_, sparse=True),
            
            nn.Embedding(sizes[1], rank*4, sparse=True),
            nn.Embedding(sizes[3], rank*3, sparse=True),
        ])

        for i in range(len(self.embeddings)):
            torch.nn.init.xavier_uniform_(self.embeddings[i].weight.data)
            
        
        self.gate1 = nn.Parameter(torch.Tensor([[0.5]]))
        self.gate2 = nn.Parameter(torch.Tensor([[0.5]]))
        
        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846


    def build_rotation(self, r):
        """
        Constructs a rotation matrix from quaternion representations.
        :param r: Quaternion tensor with shape (batch_size, rank * 4)
        :return: Rotation matrix tensor with shape (batch_size, rank, 3, 3)
        """
        r = F.normalize(r, dim=-1)
        q = r.view(r.size(0), -1, 4)

        # Initialize the rotation matrix container
        R = torch.zeros((q.size(0), q.size(1), 3, 3), device=q.device)

        w = q[:, :, 0]
        x = q[:, :, 1]
        y = q[:, :, 2]
        z = q[:, :, 3]

        # Compute the rotation matrix from quaternion components
        R[:, :, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, :, 0, 1] = 2 * (x * y - w * z)
        R[:, :, 0, 2] = 2 * (x * z + w * y)
        R[:, :, 1, 0] = 2 * (x * y + w * z)
        R[:, :, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, :, 1, 2] = 2 * (y * z - w * x)
        R[:, :, 2, 0] = 2 * (x * z - w * y)
        R[:, :, 2, 1] = 2 * (y * z + w * x)
        R[:, :, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R


    def build_scaling(self, s):
        """
        Constructs a scaling matrix, supporting both uniform and non-uniform scaling.
        :param s: Scaling factor tensor with shape (batch_size, rank) or (batch_size, rank, 3)
        :return: Scaling matrix tensor with shape (batch_size, rank, 3, 3)
        """
        # Normalize scaling factors to ensure numerical stability
        s = F.normalize(s, dim=-1)
        
        s = s.view(s.size(0), -1, 3)

        # Construct diagonal scaling matrices
        return torch.diag_embed(s)

    @staticmethod
    def has_time():
        return True
	
    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel_mean = self.embeddings[1](x[:, 1]) 
        rhs = self.embeddings[0](x[:, 2])
        time_mean = self.embeddings[2](x[:, 3])
        
        rel_rotate = self.build_rotation(self.embeddings[3](x[:, 1]))
        time_scale = self.build_scaling(self.embeddings[4](x[:, 3]))
        
        rel_time_cov = rel_rotate @ time_scale @ time_scale.transpose(2,3) @ rel_rotate.transpose(2,3)

        rel =  torch.abs(rel_mean) + rel_time_cov.view(-1,self.rank*9)
        time = torch.sin(time_mean) + rel_time_cov.view(-1,self.rank*9)       

        lhs = lhs[:, :self.half_rank_], lhs[:, self.half_rank_:]
        rel = rel[:, :self.half_rank_], rel[:, self.half_rank_:]
        rhs = rhs[:, :self.half_rank_], rhs[:, self.half_rank_:]
        time = time[:, :self.half_rank_], time[:, self.half_rank_:]


        rt = rel[0] + time[0] , rel[1] * time[1]
        return torch.sum(
            ( (lhs[0] + rt[0]) * rt[1] ) * rhs[0], 1, keepdim=True)
        
        
    def forward(self, x):

        lhs = self.embeddings[0](x[:, 0])
        rel_mean = self.embeddings[1](x[:, 1]) 
        rhs = self.embeddings[0](x[:, 2])
        time_mean = self.embeddings[2](x[:, 3])
        
        rel_rotate = self.build_rotation(self.embeddings[3](x[:, 1]))
        time_scale = self.build_scaling(self.embeddings[4](x[:, 3]))

        rel_time_cov = rel_rotate @ time_scale @ time_scale.transpose(2,3) @ rel_rotate.transpose(2,3)
        
        rel =  torch.abs(rel_mean) + rel_time_cov.view(-1,self.rank*9)
        time = torch.sin(time_mean) + rel_time_cov.view(-1,self.rank*9)       
               

        lhs = lhs[:, :self.half_rank_], lhs[:, self.half_rank_:]
        rel = rel[:, :self.half_rank_], rel[:, self.half_rank_:]
        rhs = rhs[:, :self.half_rank_], rhs[:, self.half_rank_:]
        time = time[:, :self.half_rank_], time[:, self.half_rank_:]

        right = self.embeddings[0].weight
        right = right[:, :self.half_rank_], right[:, self.half_rank_:]

        rt = rel[0] + time[0] , rel[1] * time[1]

        return (
                    ((lhs[0] + rt[0]) * rt[1] )@ right[0].t()
               ), (
                   lhs[0],
                   torch.sqrt(rt[0] ** 2 + rt[1] ** 2),
                   rhs[0]
               ),  self.embeddings[2].weight

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size][:,:self.half_rank_].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel_mean = self.embeddings[1](queries[:, 1]) 
        time_mean = self.embeddings[2](queries[:, 3])
        
        rel_rotate = self.build_rotation(self.embeddings[3](queries[:, 1]))
        time_scale = self.build_scaling(self.embeddings[4](queries[:, 3]))
      
        rel_time_cov = rel_rotate @ time_scale @ time_scale.transpose(2,3) @ rel_rotate.transpose(2,3)
        
        rel =  torch.abs(rel_mean) + rel_time_cov.view(-1,self.rank*9)
        time = torch.sin(time_mean) + rel_time_cov.view(-1,self.rank*9)       

        lhs = lhs[:, :self.half_rank_], lhs[:, self.half_rank_:]
        rel = rel[:, :self.half_rank_], rel[:, self.half_rank_:]
        time = time[:, :self.half_rank_], time[:, self.half_rank_:]

        rt = rel[0] + time[0] , rel[1] * time[1]
        
        return  (lhs[0] + rt[0]) * rt[1] 
    