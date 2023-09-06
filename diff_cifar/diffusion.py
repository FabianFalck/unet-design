import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
import math


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, multi_res_loss, sequ_train_algo, device):
        super().__init__()

        self.model = model
        self.T = T
        self.multi_res_loss = multi_res_loss
        self.sequ_train_algo = sequ_train_algo
        self.device = device

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, n_levels_used, n_downsample=0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)

        # push forward
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # model forward pass
        model_out = self.model(x_t, t, n_levels_used=n_levels_used)

        if self.multi_res_loss:
            noise_orig = noise
            # compute downsampled noise, in the same order as in decoder (coarsest to finest)
            noise_list = []  
            for k in list(range(0, self.model.n_levels))[::-1]:  # reversing the list
                
                # if using the sequential algorithm: already downsampled above, do less here.
                if self.sequ_train_algo: 
                    k = k - n_downsample

                if k > 0:  
                    xfm = DWTForward(J=k, mode='zero', wave='haar').to(self.device)
                    yl, _ = xfm(noise_orig)
                    ifm = DWTInverse(mode='zero', wave='haar').to(self.device)
                    yl_inv = ifm((yl, []))

                    # before the above lines of downsampling, noise_orig is in a certain range, and DWTForward changes the scale by a factor 2^J
                    # hence normalize yl_inv back to the original range
                    yl_inv = yl_inv / math.pow(2, k)  # correct scaling to ensure we are in the original data range

                    noise_list.append(yl_inv)
                elif k == 0: 
                    noise_list.append(noise_orig)  # on highest res, use the original noise
                # if k < 0: no noise appended

            # output
            noise = noise_list
        # else: noise is just a single tensor, i.e. no list

        loss = 0.
        loss_list = []
        if self.multi_res_loss: 
            for idx, (out, n) in enumerate(zip(model_out, noise)): 
                loss_res = F.mse_loss(out, n, reduction='none').mean()
                loss += loss_res
                loss_list.append(loss_res)
        else: 
            loss = F.mse_loss(model_out, noise, reduction='none').mean()

        return loss, loss_list


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge', multi_res_loss=False):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.multi_res_loss = multi_res_loss


        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, n_levels_used):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            if self.multi_res_loss: 
                x_prev = self.model(x_t, t, n_levels_used=n_levels_used)[-1]  # just take last output; Note: not most efficient for multi-res sampling.
            else: 
                x_prev = self.model(x_t, t, n_levels_used=n_levels_used)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            if self.multi_res_loss: 
                x_0 = self.model(x_t, t, n_levels_used=n_levels_used)[-1]  # just take last output; Note: not most efficient for multi-res sampling.
            else: 
                x_0 = self.model(x_t, t, n_levels_used=n_levels_used)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            if self.multi_res_loss: 
                eps = self.model(x_t, t, n_levels_used=n_levels_used)[-1]  # just take last output; Note: not most efficient for multi-res sampling.
            else: 
                eps = self.model(x_t, t, n_levels_used=n_levels_used)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)  

        return model_mean, model_log_var

    def forward(self, x_T, n_levels_used):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t, n_levels_used=n_levels_used)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)  
    
