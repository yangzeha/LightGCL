import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).to(device)
    result.index_add_(0, rows, col_segs)
    return result

def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score + 1e-8)
    return torch.mean(loss)

def load_msbe_neighbors(train_mat, sim_threshold=0.1):
    def get_neighbors(mat):
        mat = mat.tocsr()
        # Row normalization
        row_sums = np.array(mat.power(2).sum(axis=1)).flatten()
        row_norms = np.sqrt(row_sums)
        row_norms[row_norms == 0] = 1.0
        diag_norm = sp.diags(1.0 / row_norms)
        norm_mat = diag_norm.dot(mat)
        
        num_nodes = norm_mat.shape[0]
        neighbors = {}
        batch_size = 2000
        
        for start_idx in range(0, num_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes)
            batch_mat = norm_mat[start_idx:end_idx]
            sim_batch = batch_mat.dot(norm_mat.T)
            sim_dense = sim_batch.toarray()
            
            # Mask diagonal
            for i in range(len(sim_dense)):
                global_id = start_idx + i
                if global_id < sim_dense.shape[1]:
                    sim_dense[i, global_id] = -1.0
            
            max_indices = np.argmax(sim_dense, axis=1)
            max_values = np.max(sim_dense, axis=1)
            
            for i in range(len(sim_dense)):
                if max_values[i] > sim_threshold:
                    global_u = start_idx + i
                    best_n = max_indices[i]
                    neighbors[global_u] = [int(best_n)]
            
            if (start_idx // batch_size) % 5 == 0:
                print(f'Processed {end_idx}/{num_nodes}...')
        return neighbors

    print(f'Calculating User Neighbors (Thresh={sim_threshold})...')
    user_neighbors = get_neighbors(train_mat)
    print(f'Calculating Item Neighbors (Thresh={sim_threshold})...')
    item_neighbors = get_neighbors(train_mat.T)
    
    return user_neighbors, item_neighbors