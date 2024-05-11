import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from zbiosynth import core, layers, tasks, metrics, utils
from zbiosynth.core import Registry as R
from zbiosynth.layers import functional



@R.register("tasks.EnhancerActivityPrediction")
class EnhancerActivityPrediction(tasks.Task, core.Configurable):
    """
    CNN / DNA / enhancer activity prediction task.


    Parameters:
        model (nn.Module): CNN representation model
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

    def __init__(self, model, task=(), criterion="mse", metric=("pearsonr", "mse"), num_mlp_layer=1,
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 verbose=0):
        super(EnhancerActivityPrediction, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization
        # self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_class =  num_class
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
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "smooth_l1":
                if self.normalization:
                    loss = F.smooth_l1_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.smooth_l1_loss(pred, target, reduction="none")
            elif criterion == "l1":
                if self.normalization:
                    loss = F.l1_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.l1_loss(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean()
            
            # loss = functional.masked_mean(loss, labeled, dim=0)

            # name = tasks._get_criterion_name(criterion)
            # if self.verbose > 0:
            #     for t, l in zip(self.task, loss):
            #         metric["%s [%s]" % (name, t)] = l
            # loss = (loss * self.weight).sum() / self.weight.sum()
            metric[criterion] = loss
            all_loss += loss * weight
            
            
        return all_loss, metric
    
    def predict(self, batch, all_loss=None, metric=None):
        output = self.model(batch, all_loss=all_loss, metric=metric)
        pred = self.mlp(output["feature"])
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred
    
    def target(self, batch):
        # target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        # labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        # target[~labeled] = math.nan
        target = batch["label"].float()
        return target
    
    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)
        metric = {}
        for _metric in self.metric:
            if _metric == "mae" or _metric =="MAE":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "mse" or _metric =="MSE":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse" or _metric =="RMSE":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif _metric == "acc" or _metric =="ACC" or _metric == "accuracy":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "mcc" or _metric =="MCC" or _metric == "matthews_corrcoef":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "r2" or _metric =="R2" or _metric == "r_squared":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "spearmanr" or _metric.upper() == "SCC":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr" or _metric.upper() == "PCC":
                # score = []
                # for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                #     _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                #     score.append(_score)
                # score = torch.stack(score)           
                score = metrics.pearsonr(pred.view(-1), target.view(-1))
                score = torch.tensor(score)
            else:
                raise ValueError("This `%s` is not currently supported in this task " % _metric)
            
            # name = tasks._get_metric_name(_metric)
            # for t, s in zip(self.task, score):
            #     metric["%s [%s]" % (name, t)] = s
            
            metric[f"{_metric}"] = score.item()

        return metric