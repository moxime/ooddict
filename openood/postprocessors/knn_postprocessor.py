from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .base_postprocessor import BasePostprocessor
import os
from torch.utils.data import DataLoader
import openood.utils.comm as comm
from queue import PriorityQueue

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

def kth_largest_per_column(matrix, k):
    """
    Compute the k-th largest element for each column of a matrix.
    """
    kth_values, _ = torch.kthvalue(matrix, matrix.size(0) - k + 1, dim=0)
    return kth_values.cpu().numpy()  


def batched_matrix_multiply(ftrain, second_matrix, K, batch_size=128):
    """
    Compute the product of the feature tensors in batches.
    dimensions: ftrain: (n, d), second_matrix: (p, d), result: (n, p)
    then using the kth largest element in each column as the score
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ftrain_tensor = torch.tensor(ftrain, device=device)
    second_matrix_tensor = torch.tensor(second_matrix, device=device)
    p, _ = second_matrix_tensor.shape

    num_batches = p // batch_size + (p % batch_size > 0)
    res = []
    # for i in tqdm(range(num_batches), desc="Processing batches"):
    for i in range(num_batches):
        # Get the current batch of the second matrix
        start = i * batch_size
        end = min(start + batch_size, p)
        second_batch_tensor = second_matrix_tensor[start:end,:]
        batch_result = ftrain_tensor @ second_batch_tensor.T
        # Compute the k-th largest element for each column
        score = kth_largest_per_column(batch_result, K)
        # score = - sqrt(2(1-score))
        # score = -np.sqrt(2 * (1 - score))
        # score = score**2
        res.append(score)
    return np.concatenate(res, axis=0)


class ScoreData:
    def __init__(self, score, data):
        self.score = score
        self.data = data
    
    def __lt__(self, other):
        return float(self.score) > float(other.score)


class KNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(KNNPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.K1 = self.args.K1
        self.K2 = self.args.K2
        self.ALPHA = self.args.ALPHA
        self.queue_size = self.args.queue_size   
        self.activation_log = None
        self.id_feature = None
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False
        self.id_name = None
        self.aux_feature = None
        self.count = 0

    
    def get_auxiliary_data(self, net=None, aux_data_loader=None):
        if aux_data_loader is None:
            return None
        net.eval()
        with torch.no_grad():
            aux_feature_dict = {}
            for aux_key, aux_data_loader in aux_data_loader.items():
                aux_feature = []
                iter = 0
                for batch in tqdm(aux_data_loader,
                                        desc='get_aux_feature: ',
                                        position=0,
                                        leave=True):
                    data = batch['data'].cuda()
                    data = data[:128].float()
                    output, feature = net(data, return_feature=True)
                    aux_feature.append(
                        normalizer(feature.data.cpu().numpy()))
                    iter += 1
                    if iter==1:
                        break
                aux_feature = np.concatenate(aux_feature, axis=0)
                aux_feature_dict[aux_key] = aux_feature
            return aux_feature_dict
        

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            aux_loader = None
            for aux_key, _ in ood_loader_dict.items():
                if "t_out" == aux_key:
                    aux_loader = ood_loader_dict[aux_key]
                    break
            if aux_loader is None:
                pass
            else:
                self.aux_feature = self.get_auxiliary_data(net, aux_loader)

            # cache file folder ./cache if not exists
            if not os.path.exists('./cache'):
                os.makedirs('./cache')

            self.id_name = id_loader_dict["train"].dataset.name
            cache_name = f"cache/{self.id_name}_in_all_layers.npy"
            if not os.path.exists(cache_name):
                activation_log, msp_list = [], []
                net.eval()
                with torch.no_grad():
                    for batch in tqdm(id_loader_dict['train'],
                                      desc='Setup: ',
                                      position=0,
                                      leave=True):
                        data = batch['data'].cuda()
                        data = data.float()

                        output, feature = net(data, return_feature=True)
                        msp, _ = torch.max(torch.softmax(output, dim=1), dim=1)
                        activation_log.append(
                            normalizer(feature.data.cpu().numpy()))
                        msp_list.append(msp)
                
                msp_list = torch.cat(msp_list)
                idx = torch.argsort(msp_list, descending=True).cpu().numpy().astype(int)
                self.idx = idx


                self.activation_log = np.concatenate(activation_log, axis=0)
                self.setup_flag = True
                np.save(cache_name, self.activation_log)
                np.save(f"cache/{self.id_name}_idx.npy", self.idx)
            else:
                self.activation_log = np.load(cache_name)
                self.idx = np.load(f"cache/{self.id_name}_idx.npy")
        
                self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def acc_postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        msp, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return msp, pred, feature_normed

    @torch.no_grad()
    def conf_postprocess(self, food, ftest):
        id_train_size = self.activation_log.shape[0]    
        if id_train_size == 1281167:
            self.ALPHA = 0.5
        else:
            self.ALPHA = 0.5 # 0.5 for imagenet200
        rand_ind = np.random.choice(id_train_size, int(id_train_size * self.ALPHA), replace=False)
     
        queue_size = self.queue_size
       
        ftrain = self.activation_log[rand_ind]

        queue = PriorityQueue()

        if self.aux_feature is not None:
            # for T-out
            key = list(self.aux_feature.keys())
            if len(key)>1:
                key = key[self.count]
            else:
                key = key[0]
            self.count += 1
            prior_food = self.aux_feature[key]
            prior_score = batched_matrix_multiply(ftrain, prior_food, self.K)
            for j in range(prior_food.shape[0]):
                queue.put(ScoreData(prior_score[j], prior_food[j]))
                if queue.qsize() > queue_size:
                    queue.get()
            memory_bank = self.aux_feature[key][:5]
        else:
            memory_bank = None

    

  
        all_data = np.concatenate([ftest, food], axis=0)
        label_id = np.concatenate([np.ones(ftest.shape[0]), np.zeros(food.shape[0])], axis=0)  
        np.random.seed(100)     
        idx = np.random.permutation(all_data.shape[0])
        all_data = all_data[idx]
        label_id = label_id[idx] 

        batch_size = 512
        
        num_batches = all_data.shape[0] // batch_size + (all_data.shape[0] % batch_size > 0)
    
        scores_list = []
        
        for i in tqdm(range(num_batches), desc="Continue Learning"):
            start = i * batch_size
            end = min(start + batch_size, all_data.shape[0])
            batch_data = all_data[start:end]
            batch_score = batched_matrix_multiply(ftrain, batch_data, self.K1)
        
            for j in range(batch_score.shape[0]):
                queue.put(ScoreData(batch_score[j], batch_data[j]))
                if queue.qsize() > queue_size:
                    queue.get()
            
         
            data_list = []
            for item in list(queue.queue):
                data_list.append(item.data)
            new_food = np.array(data_list)
            if memory_bank is not None:
                new_food = np.concatenate([new_food, memory_bank], axis=0)
        
            ood_batch_score = batched_matrix_multiply(new_food, batch_data, self.K2) # 10
            batch_score = batch_score - ood_batch_score
              
            scores_list.append(batch_score)
        scores_all = np.concatenate(scores_list, axis=0)
        scores_in_final = scores_all[label_id == 1]
        scores_ood_final = scores_all[label_id == 0]

       
        return scores_in_final, scores_ood_final
    

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  dataset_name: str=None,
                  progress: bool = True):
        if dataset_name is None:
            prefix = f"cache/{data_loader.dataset.name}_vs_{self.id_name}"
        else:
            prefix = f"cache/{dataset_name}_vs_{self.id_name}"
        cache_name = prefix + "_out_all_layers.npy"
        if not os.path.exists(cache_name):
            pred_list, conf_list, label_list, feature_list = [], [], [], []
            for batch in tqdm(data_loader,
                            disable=not progress or not comm.is_main_process()):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                msp, pred, feature_normed = self.acc_postprocess(net, data)
                pred_list.append(pred.cpu())
                
                label_list.append(label.cpu())
                feature_list.append(feature_normed)
            # convert values into numpy array
            pred_list = torch.cat(pred_list).numpy().astype(int)

            label_list = torch.cat(label_list).numpy().astype(int)
            feature_list = np.concatenate(feature_list, axis=0)

            id_conf, ood_conf = self.conf_postprocess(feature_list, self.id_feature)

            np.save(cache_name, feature_list)
            np.save(prefix + "_out_label.npy", label_list)
            np.save(prefix + "_out_pred.npy", pred_list)
            return id_conf, pred_list, ood_conf, label_list
        else:
            ood_feature = np.load(cache_name)
            pred_list = np.load(prefix + "_out_pred.npy")
            label_list = np.load(prefix + "_out_label.npy")
            id_conf, ood_conf = self.conf_postprocess(ood_feature, self.id_feature)
          
            return id_conf, pred_list, ood_conf, label_list

    def acc_inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        cache_name = f"cache/{data_loader.dataset.name}_vs_{self.id_name}_out_all_layers.npy"
        if not os.path.exists(cache_name):
            pred_list, label_list, feature_list = [], [], []
            for batch in tqdm(data_loader,
                            disable=not progress or not comm.is_main_process()):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                msp, pred, feature_normed = self.acc_postprocess(net, data)
                pred_list.append(pred.cpu())
                label_list.append(label.cpu())
                feature_list.append(feature_normed)
               
            # convert values into numpy array
            pred_list = torch.cat(pred_list).numpy().astype(int)
            label_list = torch.cat(label_list).numpy().astype(int)
            feature_list = np.concatenate(feature_list, axis=0)

   
            np.save(cache_name, feature_list)
            np.save(f"cache/{data_loader.dataset.name}_vs_{self.id_name}_out_label.npy", label_list)
            np.save(f"cache/{data_loader.dataset.name}_vs_{self.id_name}_out_pred.npy", pred_list)

            self.id_feature = feature_list
      
            return pred_list, label_list
        else:
            pred_list = np.load(f"cache/{data_loader.dataset.name}_vs_{self.id_name}_out_pred.npy")
            label_list = np.load(f"cache/{data_loader.dataset.name}_vs_{self.id_name}_out_label.npy")
            self.id_feature = np.load(cache_name)
   
            return pred_list, label_list

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
