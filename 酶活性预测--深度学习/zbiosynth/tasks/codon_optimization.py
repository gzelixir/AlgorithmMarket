import math
from collections import defaultdict
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from zbiosynth import core, layers, tasks, metrics, utils
from zbiosynth.core import Registry as R
from zbiosynth.layers import functional



@R.register("tasks.CodonOptimization")
class CodonOptimization(tasks.Task, core.Configurable):
    """
    Codon Optimization task.

    Parameters:
        model (nn.Module): representation model
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
        num_class (int, optional): number of classes
        mlp_batch_norm (bool, optional): apply batch normalization in mlp or not
        mlp_dropout (float, optional): dropout in mlp
        verbose (int, optional): output verbose level
    """
    
    
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="ce", metric=("ce"), num_mlp_layer=1,
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 verbose=0):
        super(CodonOptimization, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        # self.num_class =  num_class
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.verbose = verbose
        
    def preprocess(self, train_set, valid_set, test_set):
        """
        don't need to preprocess
        """
        # weight = []
        # for task, w in self.task.items():
        #     weight.append(w)

        # self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)  ## if num_mlp_layer = 1, then hidden_dims = []
        
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [self.num_class[0]],
                            batch_norm=self.mlp_batch_norm, dropout=self.mlp_dropout)
        
    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        target = self.target(batch)
        
        # labeled = ~torch.isnan(target)
        # target[~labeled] = 0
        
        
        for criterion, weight in self.criterion.items():
            if criterion == "ce":
                # loss = F.cross_entropy(pred, target.long(), reduction="none")
                loss = F.cross_entropy(pred.view(-1, self.num_class[0]), target.view(-1))
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            
            metric[criterion] = loss
            all_loss += loss * weight

        return all_loss, metric
    
    def predict(self, batch, all_loss=None, metric=None, postprocess=False):
        '''
        Args:
            batch: batch data
            postprocess: whether to postprocess the prediction(sigmoid fucntion for binary classification)
        '''
        output = self.model(batch, all_loss=all_loss, metric=metric)
        pred = self.mlp(output["feature"])
        if self.normalization:
            pred = pred * self.std + self.mean
        if postprocess:
            # if self.criterion == 'ce':
            #     pred = pred.softmax()
            #     pred = pred.argmax(-1)
            # elif self.criterion == 'bce':
            #     pred = pred.sigmoid()
            # pred = pred.softmax()
            pred = pred.argmax(-1)
        return pred
    
    def target(self, batch):
        # target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        # labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        # target[~labeled] = math.nan
        target = batch["label"]
        return target
    
    def evaluate(self, pred, target):
        
        # print(len(pred), len(target))
        metric = {}  ## 
        if isinstance(pred, list):
            metric_tmp = {}
            for p, t in zip(pred, target):
                for _metric in self.metric:
                    if _metric == "ce":
                        score = F.cross_entropy(p.view(-1, self.num_class[0]), t.view(-1))
                    else:
                        raise ValueError("Unknown metric `%s`" % _metric)
                        
                    if f"{_metric}" not in metric_tmp:
                        metric_tmp[f"{_metric}"] = [score.item()]
                    else:
                        metric_tmp[f"{_metric}"].append(score.item())
            
            for _metric in self.metric:
                metric[f"{_metric}"] = np.mean(metric_tmp[f"{_metric}"])
                    
        else:
            for _metric in self.metric:
                if _metric == "ce":
                    score = F.cross_entropy(pred.view(-1, self.num_class[0]), target.view(-1))
                else:
                    raise ValueError("Unknown metric `%s`" % _metric)

                # if score.dim() == 0:
                #     score = score.unsqueeze(0)
                    
                # name = tasks._get_metric_name(_metric)
                metric[f"{_metric}"] = score.item()
                # for t, s in zip(self.task, score):
                #     metric["%s [%s]" % (name, t)] = s
                
        return metric