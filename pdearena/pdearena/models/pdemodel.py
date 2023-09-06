# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class
import wandb
import matplotlib.pyplot as plt
import numpy as np

from pdearena import utils
from pdearena.data.utils import PDEDataConfig
from pdearena.modules.loss import CustomMSELoss, ScaledLpLoss
from pdearena.rollout import rollout2d

from .registry import MODEL_REGISTRY

from pdearena.visualization import plot_scalar_sequence_comparison

from pdearena.modules.twod_unetbase import DWTBlock

logger = utils.get_logger(__name__)


def get_model(args, pde):
    if args.name in MODEL_REGISTRY:
        _model = MODEL_REGISTRY[args.name].copy()
        _model["init_args"].update(
            dict(
                n_input_scalar_components=pde.n_scalar_components,
                n_output_scalar_components=pde.n_scalar_components,
                n_input_vector_components=pde.n_vector_components,
                n_output_vector_components=pde.n_vector_components,
                time_history=args.time_history,
                time_future=args.time_future,
                activation=args.activation,
            )
        )
        if args.name == 'Unetbase-64_G':
            _model["init_args"].update(
                dict(
                    dwt_encoder=args.dwt_encoder,
                    up_fct = args.up_fct, 
                    n_extra_resnet_layers = args.n_extra_resnet_layers,
                    multi_res_loss = args.multi_res_loss,
                    sequ_mode = len(args.num_epochs_list) > 1,
                    hidden_channels = args.hidden_channels,
                    no_skip_connection=args.no_skip_connection, 
                    no_down_up = args.no_down_up,
                    dwt_mode = args.dwt_mode,
                    dwt_wave = args.dwt_wave,
                )
            )
        model = instantiate_class(tuple(), _model)
    else:
        logger.warning("Model not found in registry. Using fallback. Best to add your model to the registry.")
        if hasattr(args, "time_history") and args.model["init_args"]["time_history"] != args.time_history:
            logger.warning(
                f"Model time_history ({args.model['init_args']['time_history']}) does not match data time_history ({pde.time_history})."
            )
        if hasattr(args, "time_future") and args.model["init_args"]["time_future"] != args.time_future:
            logger.warning(
                f"Model time_future ({args.model['init_args']['time_future']}) does not match data time_future ({pde.time_future})."
            )
        model = instantiate_class(tuple(), args.model)

    return model


class PDEModel(LightningModule):
    def __init__(
        self,
        name: str,
        time_history: int,
        time_future: int,
        time_gap: int,
        max_num_steps: int,
        activation: str,
        criterion: str,
        lr: float,
        pdeconfig: PDEDataConfig,

        # Note: adding arguments to this method automatically adds these as parsed arguments under 'model.abc' where abc is the name of the argument.
        dwt_encoder: bool = False,
        # sequ_train_algo: bool = False,
        freeze_lower_res: bool = False,
        num_epochs_list: List[int] = [1000000000], 
        up_fct: str = 'interpolate_nearest', 
        n_extra_resnet_layers: int = 0,
        multi_res_loss: bool = False,
        hidden_channels: int = 64,
        no_skip_connection: bool = False,
        no_down_up: bool = False,
        dwt_mode: str = 'zero',
        dwt_wave: str = 'haar',

        model: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="pdeconfig")
        self.pde = pdeconfig
        if (self.pde.n_spatial_dims) == 3:
            self._mode = "3D"
        elif (self.pde.n_spatial_dims) == 2:
            self._mode = "2D"
        else:
            raise NotImplementedError(f"{self.pde}")

        self.model = get_model(self.hparams, self.pde)
        if criterion == "mse":
            self.train_criterion = CustomMSELoss()
        elif criterion == "scaledl2":
            self.train_criterion = ScaledLpLoss()
        else:
            raise NotImplementedError(f"Criterion {criterion} not implemented yet")

        self.val_criterions = {"mse": CustomMSELoss(), "scaledl2": ScaledLpLoss()}
        self.rollout_criterion = torch.nn.MSELoss(reduction="none")
        time_resolution = self.pde.trajlen
        # Max number of previous points solver can eat
        reduced_time_resolution = time_resolution - self.hparams.time_history
        # Number of future points to predict
        self.max_start_time = (
            reduced_time_resolution - self.hparams.time_future * self.hparams.max_num_steps - self.hparams.time_gap
        )

        self.dwt_encoder = self.hparams.dwt_encoder
        # self.sequ_train_algo = self.hparams.sequ_train_algo
        self.num_epochs_list = self.hparams.num_epochs_list
        self.freeze_lower_res = self.hparams.freeze_lower_res
        self.multi_res_loss = self.hparams.multi_res_loss
        self.dwt_mode = self.hparams.dwt_mode
        self.dwt_wave = self.hparams.dwt_wave
        
        self.prev_stage = 0

    def forward(self, *args):
        return self.model(*args)

    def dwt_downsample(self, x, y, n_downsample):
        with torch.no_grad(): 
            x_copy, y_copy = x.clone(), y.clone()
            # recall original dims
            x_dim_0, x_dim_1 = x.shape[0], x.shape[1]
            y_dim_0, y_dim_1 = y.shape[0], y.shape[1]
            # flatten the first two dimensions
            x = torch.flatten(x, 0, 1)
            y = torch.flatten(y, 0, 1)
            if self.multi_res_loss: 
                # downsample x completely (as in else case)
                dwt_block_x = DWTBlock(J=n_downsample, out_channels=x.shape[1], mode=self.dwt_mode, wave=self.dwt_wave).to(x.device)
                x = dwt_block_x(x)
                x = x.reshape(x_dim_0, x_dim_1, *x.shape[1:])
                # store and return all the steps of y
                y_list = []
                for j in range(n_downsample, self.model.n_levels): 
                    dwt_block_y = DWTBlock(J=j, out_channels=y.shape[1], mode=self.dwt_mode, wave=self.dwt_wave).to(x.device)
                    y_new = dwt_block_y(y)
                    y_new = y_new.reshape(y_dim_0, y_dim_1, *y_new.shape[1:])
                    if j == 0: 
                        y_list = [y_new]
                    else: 
                        y_list.append(y_new)
                # reverse the list, because the corresponding pred list will be in the order of the decoder
                y_list.reverse()  # in-place operation
                return x, y_list
            else: 
                # downsample
                # TODO it is possibly slow to create these operations every single time when downsampled --> could be sped up
                dwt_block_x = DWTBlock(J=n_downsample, out_channels=x.shape[1], mode=self.dwt_mode, wave=self.dwt_wave).to(x.device)
                dwt_block_y = DWTBlock(J=n_downsample, out_channels=y.shape[1], mode=self.dwt_mode, wave=self.dwt_wave).to(x.device)
                x = dwt_block_x(x)
                y = dwt_block_y(y)
                # reshape back
                # Note: I checked that the flatten and reshape operation keeps everything in the correct place
                # used # assert torch.equal(x, x_copy) and torch.equal(y, y_copy) to check
                x = x.reshape(x_dim_0, x_dim_1, *x.shape[1:])
                y = y.reshape(y_dim_0, y_dim_1, *y.shape[1:])
                return x, y
    
    def find_cur_stage(self):
        # find the right number of times to downsample (stage)
        num_epochs_cumsum = np.cumsum(self.num_epochs_list).tolist()
        num_epochs_cumsum = [0] + num_epochs_cumsum[:-1]  # take last value off, add 0 to front
        stage = len(self.num_epochs_list) - 1
        for cum_epoch in reversed(num_epochs_cumsum):
            if self.current_epoch >= cum_epoch: 
                break
            stage -= 1

        return int(stage)

    def freeze_layers(self, stage):
        assert self.hparams.name == "Unetbase-64_G"  # only correct for this model at the moment
        n_levels_used = stage + 1  # since we are in staged training mode
        # down
        for i in list(range(self.model.n_levels))[-n_levels_used+1:]: 
            # print("freeze down i", i)
            for param in self.model.down[i].parameters(): 
                param.grad = None
                param.requires_grad = False
        # up 
        for j in list(range(n_levels_used - 1)): 
            # print("freeze up j", j)
            for param in self.model.up[j].parameters(): 
                param.grad = None
                param.requires_grad = False
        # head
        for k in range(self.model.n_levels - n_levels_used + 1, self.model.n_levels): 
            # print("freeze head k", k)
            for param in self.model.image_proj_list[k].parameters(): 
                param.grad = None
                param.requires_grad = False
        # tail
        for l in range(n_levels_used - 1): 
            # print("freeze tail l", l)
            for param in self.model.final_list[l].parameters(): 
                param.grad = None
                param.requires_grad = False

    def compute_loss(self, pred, y):
        if self.multi_res_loss: 
            loss = 0.
            for a, b in zip(pred, y):
                loss += self.train_criterion(a, b)
        else: 
            loss = self.train_criterion(pred, y)
        return loss

    def train_step(self, batch):
        x, y = batch
        if len(self.num_epochs_list) > 1:  # sequential
            stage = self.find_cur_stage()
            # freeze layers
            if self.freeze_lower_res and stage != self.prev_stage and stage != 0: 
                self.freeze_layers(stage)
            # new stage
            if stage != self.prev_stage:
                print("New stage: ", stage, "current_epoch: ", self.current_epoch, "global_step: ", self.global_step, "---------")
                self.prev_stage = stage
        
        else: 
            # as if final stage
            stage = len(self.num_epochs_list) - 1
        # do downsample
        if len(self.num_epochs_list) > 1:  # sequential
            n_downsample = len(self.num_epochs_list) - (stage + 1)
            x, y = self.dwt_downsample(x, y, n_downsample)  # y is a list in multi_res_loss mode
        if self.hparams.name == "Unetbase-64_G":
            n_levels_used = self.model.n_levels if len(self.num_epochs_list) == 1 else stage + 1
            pred = self.model(x, n_levels_used=n_levels_used)  # pred is a list in multi_res_loss mode
        else: 
            pred = self.model(x)
        loss = self.compute_loss(pred, y)
        if self.multi_res_loss: 
            # only return highest res pred and y
            pred = pred[-1]
            y = y[-1]
        return loss, pred, y  
 
    def eval_step(self, batch):
        x, y = batch
        if len(self.num_epochs_list) > 1:  # sequential
            stage = self.find_cur_stage()
        else: 
            # as if final stage
            stage = len(self.num_epochs_list) - 1
        # do downsample
        if len(self.num_epochs_list) > 1:  # sequential
            n_downsample = len(self.num_epochs_list) - (stage + 1)
            x, y = self.dwt_downsample(x, y, n_downsample)
        if self.hparams.name == "Unetbase-64_G":
            n_levels_used = self.model.n_levels if len(self.num_epochs_list) == 1 else stage + 1
            pred = self.model(x, n_levels_used=n_levels_used)  # pred is a list in multi_res_loss mode
        else: 
            pred = self.model(x)
        loss = {k: vc(pred, y) for k, vc in self.val_criterions.items()}
        return loss, pred, y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.train_step(batch)

        if self._mode == "2D":
            scalar_loss = self.train_criterion(
                preds[:, :, 0 : self.pde.n_scalar_components, ...],
                targets[:, :, 0 : self.pde.n_scalar_components, ...],
            )

            if self.pde.n_vector_components > 0:
                vector_loss = self.train_criterion(
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )
            else:
                vector_loss = torch.tensor(0.0)
            self.log("train/loss", loss)
            self.log("train/scalar_loss", scalar_loss)
            self.log("train/vector_loss", vector_loss)
            return {
                "loss": loss,
                "scalar_loss": scalar_loss.detach(),
                "vector_loss": vector_loss.detach(),
            }
        elif self._mode == "3D":
            raise NotImplementedError(f"{self._mode}")

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        for key in outputs[0].keys():
            if "loss" in key:
                loss_vec = torch.stack([outputs[i][key] for i in range(len(outputs))])
                mean, std = utils.bootstrap(loss_vec, 64, 1)
                self.log(f"train/{key}_mean", mean)
                self.log(f"train/{key}_std", std)

    def compute_rolloutloss2D(self, batch: Any, batch_idx: int, n_levels_used):

        (u, v, cond, grid) = batch

        # print("in compute_rolloutloss2D: self.max_start_time", self.max_start_time, "self.hparams.time_future", self.hparams.time_future, 
        #       "self.hparams.time_gap", self.hparams.time_gap, "self.hparams.max_num_steps", self.hparams.max_num_steps, "self.hparams.time_history", self.hparams.time_history, 
        #       "self.pde.n_scalar_components", self.pde.n_scalar_components, "self.pde.n_vector_components", self.pde.n_vector_components)
        
            #   "self.pde.trajlen", self.pde.trajlen, "self.pde.n_scalar_components", self.pde.n_scalar_components, "self.pde.n_vector_components", self.pde.n_vector_components, 
            #   "self.pde.n_spatial_dims", self.pde.n_spatial_dims, "self.pde.n_dims", self.pde.n_dims, "self.pde.n_components", self.pde.n_components, "self.pde.n_cond", self.pde.n_cond)

        losses = []
        for start in range(
            0,
            self.max_start_time + 1,
            self.hparams.time_future + self.hparams.time_gap,
        ):

            end_time = start + self.hparams.time_history
            target_start_time = end_time + self.hparams.time_gap
            target_end_time = target_start_time + self.hparams.time_future * self.hparams.max_num_steps

            init_u = u[:, start:end_time, ...]
            if self.pde.n_vector_components > 0:
                init_v = v[:, start:end_time, ...]
            else:
                init_v = None

            pred_traj = rollout2d(
                self.model,
                init_u,
                init_v,
                grid,
                self.pde,
                self.hparams.time_history,
                self.hparams.max_num_steps,
                n_levels_used
            )
            targ_u = u[:, target_start_time:target_end_time, ...]
            if self.pde.n_vector_components > 0:
                targ_v = v[:, target_start_time:target_end_time, ...]
                targ_traj = torch.cat((targ_u, targ_v), dim=2)
            else:
                targ_traj = targ_u
            # print("init_u", init_u.shape, "init_v", init_v.shape, "pred_traj", pred_traj.shape, "targ_traj", targ_traj.shape)

            # plot the trajectory
            # take first batch (0th dimension)
            # take only scalar (2nd dimension)
            if start == 0 and batch_idx in [1,2,3]:  # only on first time step; and on the first three batches
                fig = plot_scalar_sequence_comparison(init_field=init_u[0,:,0,:,:].detach().cpu().numpy(), ground_truth=targ_traj[0,:,0,:,:].detach().cpu().numpy(), prediction=pred_traj[0,:,0,:,:].detach().cpu().numpy())  # init_u torch.Size([8, 4, 1, 128, 128]) init_v torch.Size([8, 4, 2, 128, 128]) pred_traj torch.Size([8, 5, 3, 128, 128]) targ_traj torch.Size([8, 5, 3, 128, 128])
                # log the fig
                self.logger.experiment.log({"valid/plot/scalar_sequence_comparison": [wandb.Image(plt)]})  # see https://github.com/Lightning-AI/lightning/issues/3635 why I use this call
                plt.close(fig=fig)

            loss = self.rollout_criterion(pred_traj, targ_traj).mean(dim=(0, 2, 3, 4))
            losses.append(loss)
        loss_vec = torch.stack(losses, dim=0).mean(dim=0)
        return loss_vec

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # TODO there is a bug here: batch is int, but should be tensors which are in batch_idx
        # TODO fix in principled way !!!!!!!!!!!!!!!!!!!
        # if type(batch) == int: 
        #     id = batch
        #     batch = batch_idx
        #     batch_idx = id

        # print("validation step")

        if dataloader_idx == 0:
            # one-step loss
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "2D":
                loss["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.pde.n_scalar_components, ...],
                    targets[:, :, 0 : self.pde.n_scalar_components, ...],
                )
                loss["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )

                for k in loss.keys():
                    self.log(f"valid/loss/{k}", loss[k])
                return {f"{k}_loss": v for k, v in loss.items()}

            elif self._mode == "3D":
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            # rollout loss
            if self._mode == "2D":
                # check if self.model has attribute n_levels
                if hasattr(self.model, "n_levels"):
                    n_levels_used = self.model.n_levels if len(self.num_epochs_list) == 1 else self.prev_stage + 1
                else: 
                    n_levels_used = None
                loss_vec = self.compute_rolloutloss2D(batch, batch_idx, n_levels_used=n_levels_used)
            elif self._mode == "3D":
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss = loss_vec.sum()
            loss_t = loss_vec.cumsum(0)
            chan_avg_loss = loss / (self.pde.n_scalar_components + self.pde.n_vector_components)
            self.log("valid/unrolled_loss", loss)
            return {
                "unrolled_loss": loss,
                "loss_timesteps": loss_t,
                "unrolled_chan_avg_loss": chan_avg_loss,
            }

    def validation_epoch_end(self, outputs: List[Any]):
        if len(outputs) > 1:
            if len(outputs[0]) > 0:
                for key in outputs[0][0].keys():
                    if "loss" in key:
                        loss_vec = torch.stack([outputs[0][i][key] for i in range(len(outputs[0]))])
                        mean, std = utils.bootstrap(loss_vec, 64, 1)
                        self.log(f"valid/{key}_mean", mean)
                        self.log(f"valid/{key}_std", std)

            if len(outputs[1]) > 0:
                unrolled_loss = torch.stack([outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))])
                loss_timesteps_B = torch.stack([outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))])
                loss_timesteps = loss_timesteps_B.mean(0)

                for i in range(self.hparams.max_num_steps):
                    self.log(f"valid/intime_{i}_loss", loss_timesteps[i])

                mean, std = utils.bootstrap(unrolled_loss, 64, 1)
                self.log("valid/unrolled_loss_mean", mean)
                self.log("valid/unrolled_loss_std", std)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            loss, preds, targets = self.eval_step(batch)
            if self._mode == "2D":
                loss["scalar_mse"] = self.val_criterions["mse"](
                    preds[:, :, 0 : self.pde.n_scalar_components, ...],
                    targets[:, :, 0 : self.pde.n_scalar_components, ...],
                )
                loss["vector_mse"] = self.val_criterions["mse"](
                    preds[:, :, self.pde.n_scalar_components :, ...],
                    targets[:, :, self.pde.n_scalar_components :, ...],
                )

                self.log("test/loss", loss)
                return {f"{k}_loss": v for k, v in loss.items()}
            elif self._mode == "3D":
                raise NotImplementedError(f"{self._mode}")

        elif dataloader_idx == 1:
            if self._mode == "2D":
                # check if self.model has attribute n_levels
                if hasattr(self.model, "n_levels"):
                    n_levels_used = self.model.n_levels if len(self.num_epochs_list) == 1 else self.prev_stage + 1
                else: 
                    n_levels_used = None
                loss_vec = self.compute_rolloutloss2D(batch, batch_idx=batch_idx, n_levels_used=n_levels_used)
            elif self._mode == "3D":
                raise NotImplementedError(f"{self._mode}")
            # summing across "time axis"
            loss = loss_vec.sum()
            loss_t = loss_vec.cumsum(0)
            self.log("test/unrolled_loss", loss)
            # self.log("valid/normalized_unrolled_loss", loss)
            return {
                "unrolled_loss": loss,
                "loss_timesteps": loss_t,
            }

    def test_epoch_end(self, outputs: List[Any]):
        assert len(outputs) > 1
        if len(outputs[0]) > 0:
            for key in outputs[0][0].keys():
                if "loss" in key:
                    loss_vec = torch.stack([outputs[0][i][key] for i in range(len(outputs[0]))])
                    mean, std = utils.bootstrap(loss_vec, 64, 1)
                    self.log(f"test/{key}_mean", mean)
                    self.log(f"test/{key}_std", std)
        if len(outputs[1]) > 0:
            unrolled_loss = torch.stack([outputs[1][i]["unrolled_loss"] for i in range(len(outputs[1]))])
            loss_timesteps_B = torch.stack([outputs[1][i]["loss_timesteps"] for i in range(len(outputs[1]))])
            loss_timesteps = loss_timesteps_B.mean(0)
            for i in range(self.hparams.max_num_steps):
                self.log(f"test/intime_{i}_loss", loss_timesteps[i])

            mean, std = utils.bootstrap(unrolled_loss, 64, 1)
            self.log("test/unrolled_loss_mean", mean)
            self.log("test/unrolled_loss_std", std)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
