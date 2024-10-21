from collections import defaultdict
from aeon.datasets import load_classification
from aeon.clustering.averaging import elastic_barycenter_average
import numpy as np
import pandas as pd
import torch
import wandb
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.barycenters import dtw_barycenter_averaging
from sklearn.preprocessing import LabelEncoder

# Assuming background_data is a numpy array with shape (num_samples, num_time_steps)
from nte.experiment.utils import tv_norm, save_timeseries
from nte.models.saliency_model import Saliency


class CFExplainer_ShapeDBA(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(CFExplainer_ShapeDBA, self).__init__(background_data=background_data,
                                            background_label=background_label,
                                            predict_fn=predict_fn,
                                            )
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.conf_threshold = 0.8
        self.eps = None
        self.eps_decay = 0.9991

    def shape_dba_retrieval(self, target_label):
        X, y = load_classification(name="TwoLeadECG", split="train")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        shapedba = elastic_barycenter_average(X[y == target_label], distance="shape_dtw", reach=15)

        shapedba = shapedba.reshape(-1)
        return shapedba

    def cf_label_fun(self, instance):
        # print("cf_label_fun：", instance.shape)
        output = self.softmax_fn(self.predict_fn(instance.reshape(1, 1, instance.shape[0]).float()))
        target = torch.argsort(output, descending=True)[0, 1].item()
        return target

    def generate_saliency(self, data, label, **kwargs):
        self.mode = 'Explore'
        # print(data.shape)

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        top_prediction_class = np.argmax(kwargs['target'].cpu().data.numpy())

        cf_label = self.cf_label_fun(data)

        rep = self.shape_dba_retrieval(cf_label)
        # barycenter_tensor = torch.tensor(barycenter, dtype=torch.float32)

        self.eps = 1.0

        mask_init = np.random.uniform(size=[1, data.shape[-1]], low=0, high=1)
        mask = Variable(torch.from_numpy(mask_init), requires_grad=True)
        assert mask.shape.numel() == 1 * data.shape[-1]

        # Setup optimizer
        optimizer = torch.optim.Adam([mask], lr=self.args.lr)

        if self.args.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

        print(f"{self.args.algo}: Optimizing... ")
        metrics = defaultdict(lambda: [])


        max_iterations_without_improvement = 30  # Define the maximum number of iterations without improvement
        imp_threshold = 0.001
        # cf_prob = float('-inf')
        best_loss = float('inf')  # Track the best 'Confidence' achieved
        counter = 0  # Counter for iterations without improvement

        # Training
        i = 0
        while i <= self.args.max_itr:
            Rt = torch.tensor(rep, dtype=torch.float32)
            # print(Rt.shape, data.shape, mask.shape)

            perturbated_input = data.mul(1-mask) + Rt.mul(mask)
            # print(perturbated_input.shape, mask.shape)
            assert perturbated_input.shape == mask.shape

            pred_outputs = self.softmax_fn(self.predict_fn(perturbated_input.reshape(1, 1, perturbated_input.shape[1]).float()))

            l_maximize = 1 - pred_outputs[0][cf_label]
            l_budget_loss = torch.mean(torch.abs(mask)) * float(self.args.enable_budget)
            l_tv_norm_loss = tv_norm(mask, self.args.tv_beta) * float(self.args.enable_tvnorm)
            # l_l1_loss = torch.mean(torch.abs(perturbated_input - data))
            # l_smoothness = torch.mean(torch.abs(perturbated_input[:, 1:] - perturbated_input[:, :-1]))

            loss = (self.args.l_budget_coeff * l_budget_loss) + \
                   (self.args.l_tv_norm_coeff * l_tv_norm_loss) + \
                   (self.args.l_max_coeff * l_maximize)


            if best_loss - loss < imp_threshold:
                counter += 1
            else:
                counter = 0  # Reset counter if there is an improvement
                best_loss = loss  # Update the best 'Confidence' achieved

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            # mask.grad.data.clamp_(0, 1)

            metrics['Mean Gradient'].append(float(np.mean(mask.grad.cpu().detach().numpy())))
            metrics['L_Maximize'].append(float(l_maximize.item()))
            metrics['L_Budget'].append(float(l_budget_loss.item()))
            metrics['L_TV_Norm'].append(float(l_tv_norm_loss.item()))
            # metrics['L_L1'].append(float(l_l1_loss.item()))
            # metrics['L_Smooth'].append(float(l_smoothness.item()))
            metrics['L_Total'].append(float(loss.item()))
            metrics['CF_Prob'].append(float(pred_outputs[0][cf_label].item()))

            optimizer.step()

            if self.args.enable_lr_decay:
                scheduler.step(epoch=i)

            # Clamp mask
            mask.data.clamp_(0, 1)

            if self.enable_wandb:
                _mets = {**{k: v[-1] for k, v in metrics.items() if k != "epoch"}}
                if f"epoch_{i}" in metrics["epoch"]:
                    _mets = {**_mets, **metrics["epoch"][f"epoch_{i}"]['eval_metrics']}
                wandb.log(_mets)

            # Check if early stopping condition is met
            if counter >= max_iterations_without_improvement:
                print("Early stopping triggered: 'total loss' metric didn't improve much")
                break
            else:
                i += 1


        mask = mask.cpu().detach().numpy().flatten()
        mask = (mask-mask.min())/(mask.max()-mask.min())

        target_prob = float(pred_outputs[0][cf_label].item())

        save_timeseries(mask=mask, raw_mask=None,
                        time_series=data.numpy(),
                        perturbated_output=perturbated_input,
                        save_dir=kwargs['save_dir'],
                        enable_wandb=self.enable_wandb, algo=self.args.algo,
                        dataset=self.args.dataset,
                        category=top_prediction_class)


        return mask, perturbated_input, target_prob



