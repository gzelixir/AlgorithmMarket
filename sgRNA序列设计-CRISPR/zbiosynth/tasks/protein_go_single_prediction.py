import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from zbiosynth import core, layers, tasks, metrics, utils
from zbiosynth.core import Registry as R
from zbiosynth.layers import functional



@R.register("tasks.ProteinGOSinglePrediction")
class ProteinGOSinglePrediction(tasks.Task, core.Configurable):
    """
    Protein GO single task [BP, MF, CC] prediction task.

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

    def __init__(self, model, task=(), criterion="bce", metric=("acc", "bce"), num_mlp_layer=1,
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 verbose=0):
        super(ProteinGOSinglePrediction, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization
        self.num_class = num_class
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
        
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [self.num_class],
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
            losses = []
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            
            loss = loss.mean()
            
            # name = tasks._get_criterion_name(criterion)
            # if self.verbose > 0:
            #     for t, l in zip(self.task, loss):
            #         metric["%s [%s]" % (name, t)] = l
            # loss = (loss * self.weight).sum() / self.weight.sum()
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
        
        if postprocess:
            pred = pred.sigmoid()
            
        return pred
    
    def target(self, batch):
        # target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        # labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        # target[~labeled] = math.nan
        target = batch["label"].float()
        return target
    
    def evaluate(self, pred, target):

        metric = {}
        for _metric in self.metric:
            scores = []
            if _metric == "bce":
                score = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                score = score.mean()
            elif _metric == 'fmax':
                threshold,_,_, score = metrics.fmax_score(target, pred)
                score = torch.tensor(score)
            # elif _metric == "acc" or _metric =="ACC" or _metric == "accuracy":
            #     if 'bce' in self.criterion:
            #         probs = torch.sigmoid(pred)
            #         score = torch.sum((probs[labeled] > 0.5) == (target[labeled])) / len(target)
            #     else:
            #         score = pred[labeled].argmax(-1) == target[labeled]
            #         score = score.sum() / target.size(0)
            # elif _metric == 'precision':
            #     if 'bce' in self.criterion:
            #         probs = torch.sigmoid(pred)
            #         score = metrics.pre_score((probs[labeled] > 0.5).long().cpu(), target[labeled].long().cpu())
            #         score = torch.tensor(score)
            # elif _metric == 'b_acc' or _metric == 'balanced_accuracy':
            #     if 'bce' in self.criterion:
            #         probs = torch.sigmoid(pred)
            #         score = metrics.balanced_accuracy((probs[labeled] > 0.5).long().cpu(), target[labeled].long().cpu())
            #         score = torch.tensor(score)
            # elif _metric == "mcc" or _metric =="MCC" or _metric == "matthews_corrcoef":
            #     score = []
            #     if "bce" in self.criterion:
            #         probs = torch.sigmoid(pred)
            #         # score,_,_,_,_ = metrics.mcc_score((probs[labeled] > 0.5).long().cpu(), target[labeled].long().cpu())
            #         score = metrics.mcc_score((probs[labeled] > 0.5).long().cpu(), target[labeled].long().cpu())
            #         score = torch.tensor(score)
            #     else:   
            #         num_class = 0
            #         for i, cur_num_class in enumerate(self.num_class):
            #             _pred = pred[:, num_class:num_class + cur_num_class]
            #             _target = target[:, i]
            #             _labeled = labeled[:, i]
            #             _score = metrics.matthews_corrcoef(_pred[_labeled].cpu(), _target[_labeled].long().cpu())
            #             score.append(_score)
            #             num_class += cur_num_class
            #         score = torch.stack(score)
            # elif _metric == "auroc" or _metric =="AUROC" or _metric == "roc_auc_score":
            #     if 'bce' in self.criterion:
            #         probs = torch.sigmoid(pred)
            #         score = metrics.area_under_roc(probs[labeled], target[labeled])
            # elif _metric == "f1" or _metric == "F1":
            #     probs = torch.sigmoid(pred)
            #     score = metrics.f1_score((probs[labeled] > 0.5).long().cpu(), target[labeled].long().cpu())
            #     score = torch.tensor(score)
            else:
                raise ValueError("This `%s` is not currently supported in this task " % _metric)
                

            metric[f"{_metric}"] = score.item()
            if _metric in ['fmax']:
                metric[f"{_metric}_threshold"] = threshold
                
        return metric