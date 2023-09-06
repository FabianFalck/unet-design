import torch
import numpy as np
from torch_ddpm.ddpm.utils import batch_mul, DataClass
from torch_ddpm.ddpm.typing import Function


def get_mean_scale_reverse_fn(
    score_fn: Function,
    mean_scale_fn: Function,
    eps: float = 1e-3,
    T: float = 1.0,
    N: int = 1000,
    n_levels_used: int = -1, 
):
    def simulate_reverse_diffusion(x_T: torch.Tensor):
        shape = x_T.shape
        B = shape[0]

        timesteps = torch.linspace(T, eps, N).to(x_T.device)  # where T and epsilon are used

        def loop_body(i, val):
            # print("loop body", i, val[0].shape)
            x, x_mean = val
            t = timesteps[i]
            vec_t = (torch.ones(B).to(x_T.device) * t).reshape(B, 1)
            x_mean, scale = mean_scale_fn(x, vec_t, score_fn, n_levels_used=n_levels_used)
            noise = torch.randn(x.shape).to(x_T.device)
            x = x_mean + batch_mul(scale, noise)
            return x, x_mean

        loop_state = (x_T, x_T)
        for i in range(N):
            loop_state = loop_body(i, loop_state)
        x, x_mean = loop_state

        return x, x_mean

    return simulate_reverse_diffusion


class Diffusion(torch.nn.Module):
    """ """

    def __init__(self, beta_min=0.1, beta_max=20, N=1000, eps=1e-3, T=1.0, multi_res_loss=False, weighted_multi_res_loss=False) -> None:
        super().__init__()
        self.register_buffer("beta_0", torch.tensor(beta_min))
        self.register_buffer("beta_1", torch.tensor(beta_max))
        self.N = N  # number of timesteps - 1 (index in fixed parameter lists, one-off from # time steps)
        self.T = T  # TODO is this temperature?
        self.eps = eps
        self.multi_res_loss = multi_res_loss
        self.weighted_multi_res_loss = weighted_multi_res_loss

        # beta_max, together with "alphas =" line below, needs to be < N
        discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)  # to gpu
        # register buffer: parameters which are not returned by .parameters() and hence not changed by the optimizer, see https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        self.register_buffer("discrete_betas", discrete_betas)

        alphas = 1.0 - self.discrete_betas
        self.register_buffer("alphas", alphas)

        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.register_buffer("sqrt_1m_alphas_cumprod", sqrt_1m_alphas_cumprod)

    def sample_t(self, x_0: torch.Tensor, stage = None, n_stages = None) -> torch.Tensor:
        # return Categorical(probs=torch.ones(self.N)).sample((x_0.shape[0],))
        if stage is not None:
            assert n_stages is not None
            # another alternative would be to have non-overlapping time windows, but this would require further changes
            N_min = int(self.N * ((n_stages - stage - 1) / n_stages))
            timesteps = np.arange(start=N_min, stop=self.N)  # [.6, 1.], [.3, 1.] etc.
            indices_np = np.random.choice(timesteps, size=(x_0.shape[0],))
            indices = torch.from_numpy(indices_np).long().to(x_0.device)
        else: 
            indices_np = np.random.choice(self.N, size=(x_0.shape[0],))
            indices = torch.from_numpy(indices_np).long().to(x_0.device)

        return indices

    def sample_x(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x_0)
        # something like (10) in https://arxiv.org/pdf/2006.11239.pdf   
        # Comment: this is just computing x_t(x_0, eps), as given above Eq. (9)
        x_t = batch_mul(self.sqrt_alphas_cumprod[t], x_0) + batch_mul(
            self.sqrt_1m_alphas_cumprod[t], noise
        )

        return DataClass(x_t=x_t, z=noise, t=t)

    
    def loss(self, model_output: torch.Tensor, noise: torch.Tensor, last_loss_schedule_weight: float) -> torch.Tensor:
        # Version B: multi-output loss for unet wavelet
        if self.multi_res_loss: 
            loss = 0.
            loss_list = []

            if self.multi_res_loss and self.weighted_multi_res_loss: 
                weight_list = []
                for out in model_output: 
                    # weight is inverse proporrtional to the number of pixels on the resolution
                    # this downweights the finer resolutions
                    weight_list.append(1 / (out.shape[2]^2))  
                weight = np.array(weight_list)
                weight = weight / np.sum(weight)  # normalize; this will make the weights change during sequential training
                weights = weight.tolist()
            else: 
                weights = [1.] * len(model_output)

            # model_output and noise is list, from coarsest to finest (decoder order)
            for i, (out, n) in enumerate(zip(model_output, noise)): 
                losses = torch.square(out - n)
                losses = torch.mean(losses.reshape((losses.shape[0], -1)), axis=-1)
                loss_res = torch.mean(losses)
                if i == len(model_output) - 1: 
                    loss += loss_res * weights[i] * last_loss_schedule_weight
                else: 
                    loss += loss_res * weights[i]
                loss_list.append(loss_res)

        # Version A: original loss    
        else: 
            losses = torch.square(model_output - noise)
            losses = torch.mean(losses.reshape((losses.shape[0], -1)), axis=-1)
            loss = torch.mean(losses)
            # dummy loss_list (not used here)
            loss_list = []

        return loss, loss_list

    def reverse_mean_scale_function(
        self, x_t: torch.Tensor, t: torch.Tensor, score_fn: Function, n_levels_used: int = -1
    ) -> torch.Tensor:
        timestep = t * (self.N - 1) / self.T
        t_label = timestep.type(torch.int64)
        beta = self.discrete_betas[t_label]

        model_pred, _ = score_fn(x_t, timestep, n_levels_used=n_levels_used)

        if self.multi_res_loss:
            model_pred = model_pred[-1]

        std = self.sqrt_1m_alphas_cumprod[t_label.type(torch.int64)]
        score = -batch_mul(model_pred, 1.0 / std)
        x_mean = batch_mul((x_t + batch_mul(beta, score)), 1.0 / torch.sqrt(1.0 - beta))
        return x_mean, torch.sqrt(beta)

    def reverse_sample(self, x_T: torch.Tensor, score_fn: Function, n_levels_used: int = -1):
        sample_fn = get_mean_scale_reverse_fn(
            score_fn=score_fn,
            mean_scale_fn=self.reverse_mean_scale_function,
            N=self.N,
            T=self.T,
            n_levels_used=n_levels_used,
        )
        with torch.no_grad():
            return sample_fn(x_T)


    def reverse_sample_partly(self, x_T: torch.Tensor, score_fn: Function, N: int, T: float, eps: float):
        sample_fn = get_mean_scale_reverse_fn(
            score_fn=score_fn,
            mean_scale_fn=self.reverse_mean_scale_function,
            N=N,
            T=T,
            eps=eps,
        )
        with torch.no_grad():
            return sample_fn(x_T)
