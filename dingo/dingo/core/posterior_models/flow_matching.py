import torch
from torch import nn
import math
from .cflow_base import ContinuousFlowsBase, ContinuousFlowsEuler, compute_log_prior


class FlowMatching(ContinuousFlowsBase):
    """
    Class for continuous normalizing flows trained with flow matching.

        t               ~ U[0, 1-eps)                               noise level
        theta_0         ~ N(0, 1)                                   sampled noise
        theta_1         = theta                                     pure sample
        theta_t         = c1(t) * theta_1 + c0(t) * theta_0         noisy sample

        eps             = 0
        c0              = (1 - (1 - sigma_min) * t)
        c1              = t

        v_target        = theta_1 - (1 - sigma_min) * theta_0
        loss            = || v_target - network(theta_t, t) ||
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eps = 0
        self.sigma_min = self.model_kwargs["posterior_kwargs"]["sigma_min"]

    def evaluate_vectorfield(self, t, theta_t, *context_data):
        """
        Vectorfield that generates the flow, see Docstring in ContinuousFlowsBase for
        details. With flow matching, this vectorfield is learnt directly.
        """
        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=theta_t.device)
        return self.network(t, theta_t, *context_data)

    def loss(self, theta, *context_data):
        """
        Calculates loss as the the mean squared error between the predicted vectorfield
        and the vector field for transporting the parameter data to samples from the
        prior.

        Parameters
        ----------
        theta: torch.tensor
            parameters (e.g., binary-black hole parameters)
        *context_data: list[torch.Tensor]
            context data (e.g., gravitational-wave data)

        Returns
        -------
        torch.tensor
            loss tensor
        """
        # Shall we allow for multiple time evaluations for every data, context pair (to improve efficiency)?
        mse = nn.MSELoss()

        t = self.sample_t(len(theta))
        theta_0 = self.sample_theta_0(len(theta))
        theta_1 = theta
        theta_t = ot_conditional_flow(theta_0, theta_1, t, self.sigma_min)
        true_vf = theta - (1 - self.sigma_min) * theta_0

        predicted_vf = self.network(t, theta_t, *context_data)
        loss = mse(predicted_vf, true_vf)
        return loss

class FlowMatchingEuler(ContinuousFlowsEuler):
    """
    Class for continuous normalizing flows trained with flow matching.

        t               ~ U[0, 1-eps)                               noise level
        theta_0         ~ N(0, 1)                                   sampled noise
        theta_1         = theta                                     pure sample
        theta_t         = c1(t) * theta_1 + c0(t) * theta_0         noisy sample

        eps             = 0
        c0              = (1 - (1 - sigma_min) * t)
        c1              = t

        v_target        = theta_1 - (1 - sigma_min) * theta_0
        loss            = || v_target - network(theta_t, t) ||
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eps = 0
        self.sigma_min = self.model_kwargs["posterior_kwargs"]["sigma_min"]

    def evaluate_vectorfield(self, t, theta_t, *context_data):
        """
        Vectorfield that generates the flow, see Docstring in ContinuousFlowsBase for
        details. With flow matching, this vectorfield is learnt directly.
        """
        # If t is a number (and thus the same for each element in this batch),
        # expand it as a tensor. This is required for the odeint solver.
        t = t * torch.ones(len(theta_t), device=theta_t.device)
        return self.network(t, theta_t, *context_data)

    def loss(self, theta, *context_data):
        """
        Calculates loss as the the mean squared error between the predicted vectorfield
        and the vector field for transporting the parameter data to samples from the
        prior.

        Parameters
        ----------
        theta: torch.tensor
            parameters (e.g., binary-black hole parameters)
        *context_data: list[torch.Tensor]
            context data (e.g., gravitational-wave data)

        Returns
        -------
        torch.tensor
            loss tensor
        """
        # Shall we allow for multiple time evaluations for every data, context pair (to improve efficiency)?
        mse = nn.MSELoss()

        t = self.sample_t(len(theta))
        theta_0 = self.sample_theta_0(len(theta))
        theta_1 = theta
        theta_t = ot_conditional_flow(theta_0, theta_1, t, self.sigma_min)
        true_vf = theta - (1 - self.sigma_min) * theta_0

        predicted_vf = self.network(t, theta_t, *context_data)
        loss = mse(predicted_vf, true_vf)
        return loss

class FlowMatching_Augmentation(FlowMatchingEuler):
    """
    Class for continuous normalizing flows trained with flow matching.

        t               ~ U[0, 1-eps)                               noise level
        theta_0         ~ N(0, 1)                                   sampled noise
        theta_1         = theta                                     pure sample
        theta_t         = c1(t) * theta_1 + c0(t) * theta_0         noisy sample

        eps             = 0
        c0              = (1 - (1 - sigma_min) * t)
        c1              = t

        v_target        = theta_1 - (1 - sigma_min) * theta_0
        loss            = || v_target - network(theta_t, t) ||
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eps = 0
        self.sigma_min = self.model_kwargs["posterior_kwargs"]["sigma_min"]
        self.repeated_batch_size = self.model_kwargs["posterior_kwargs"].get("repeated_batch_size", 1)

    def loss(self, theta, *context_data):
        """
        Calculates loss as the the mean squared error between the predicted vectorfield
        and the vector field for transporting the parameter data to samples from the
        prior.

        Parameters
        ----------
        theta: torch.tensor
            parameters (e.g., binary-black hole parameters)
        *context_data: list[torch.Tensor]
            context data (e.g., gravitational-wave data)

        Returns
        -------
        torch.tensor
            loss tensor
        """
        # Shall we allow for multiple time evaluations for every data, context pair (to improve efficiency)?
        mse = nn.MSELoss()

        # repeat each data point repeated_batch_size times
        t = self.sample_t(len(theta) * self.repeated_batch_size)
        theta_0 = self.sample_theta_0(len(theta))
        theta_0 = theta_0.repeat_interleave(self.repeated_batch_size, dim=0)
        theta = theta.repeat_interleave(self.repeated_batch_size, dim=0)
        theta_1 = theta
        theta_t = ot_conditional_flow(theta_0, theta_1, t, self.sigma_min)

        context_data = [cd.repeat_interleave(self.repeated_batch_size, dim=0) for cd in context_data]

        true_vf = theta - (1 - self.sigma_min) * theta_0
        predicted_vf = self.network(t, theta_t, *context_data)

        loss = mse(predicted_vf, true_vf)
        return loss

class FlowMatchingV2_with_v_pred_mcmc(FlowMatching_Augmentation):
    """
    Class for continuous normalizing flows trained with flow matching.
    Includes built-in SMC/MCMC guided sampling using Bilby likelihoods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # @torch.no_grad()
    # def sample(self, *context_data, num_samples: int = None, _score_fn=None, config=None):
    #     """
    #     Main sampling method (SDE + SMC Guidance), samples ONLY.
    #     Maintains explicit [Batch, Particles, Dim] shape.
    #     """
        
    #     # ==================================================================
    #     # 1. 配置 & 维度检查
    #     # ==================================================================
    #     if len(context_data) > 0:
    #         input_batch_size = context_data[0].shape[0] * num_samples
    #     else:
    #         raise ValueError("Context data required for guided sampling.")

    #     if config == None:
    #         beam_width = 4          
    #         branch_factor = 4     
    #         alpha_before = 0.0
    #         alpha_on = 1.0
    #         alpha_after = 0.0     
    #         prune_window = (0.1, 0.95)   
    #         score_every_prune = 2       
    #         select_mode = "softmax_resample" 
    #         temperature = 1.0           
    #         output_mode = "theta1_pred" 
    #     else:
    #         beam_width = config["beam_width"]
    #         branch_factor = config["branch_factor"]
    #         alpha_before = config["alpha_before"]
    #         alpha_on = config["alpha_on"]
    #         alpha_after = config["alpha_after"]
    #         prune_start_fraction = config["prune_start_fraction"]
    #         prune_end_fraction = config["prune_end_fraction"]
    #         score_every_prune = config["score_every_prune"]
    #         select_mode = config["select_mode"]
    #         temperature = config["temperature"]
    #         output_mode = config["output_mode"]
    #         prune_window = (prune_start_fraction, prune_end_fraction)
        
    #     n_candidates_full = beam_width * branch_factor 

    #     print(f"Starting Guided Sampling (Samples Only). Batch: {input_batch_size}, Beam: {beam_width}, Branch: {branch_factor}")

    #     def _step_update(theta, u, t, context_list, alpha = 0.0):
    #         """
    #         Input:  theta [B, P, D]
    #         Output: theta [B, P, D], theta1_pred [B, P, D]
    #         """
    #         B, P, D = theta.shape
    #         current_total = B * P
            
    #         # 1. Flatten for Network
    #         theta_flat = theta.view(current_total, D)
            
    #         t_vec = t.to(theta.device, theta.dtype).view(1).expand(current_total)
    #         u_vec = u.to(theta.device, theta.dtype).view(1).expand(current_total)
    #         dt = t_vec - u_vec 

    #         # 2. Network Calculation (No Divergence needed)
    #         # context_list is already expanded to [B*P, ...]
    #         # evaluate_vectorfield returns [B*P, D]
    #         predicted_vf = self.evaluate_vectorfield(u_vec, theta_flat, *context_list)
            
    #         # 3. Predict theta_1 (Flat)
    #         theta_1_pred_flat = theta_flat + (1 - u_vec.unsqueeze(1)) * predicted_vf

    #         # 4. SDE Update (Flat)
    #         safe_t = torch.clamp(u_vec, min=1e-5).unsqueeze(1)
    #         sigma_t = alpha * torch.sqrt((1 - safe_t) / (safe_t + 0.1))
    #         correction = (sigma_t**2 / (2 * (1 - safe_t))) * (safe_t * predicted_vf - theta_flat)
    #         drift_theta = (predicted_vf + correction) * dt.unsqueeze(1)
    #         diffusion_theta = sigma_t * torch.sqrt(dt.unsqueeze(1)) * torch.randn_like(theta_flat)
    #         theta_next_flat = theta_flat + drift_theta + diffusion_theta
            
    #         # 5. Unflatten -> [B, P, D]
    #         theta_next = theta_next_flat.view(B, P, D)
    #         # theta_1_pred = theta_1_pred_flat.view(B, P, D)

    #         return theta_next, theta_1_pred_flat

    #     def _select_and_expand(theta_next, theta1_pred, scores_flat, final = False):
    #         """
    #         Input: [Batch, Current_Particles, D]
    #         Output: [Batch, Beam * Branch, D]
    #         """
    #         B, P, D = theta_next.shape
            
    #         # 1. Reshape Scores [B, P]
    #         # scores_flat comes from _score_fn which returns [B*P]
    #         scores_view = scores_flat.view(B, P)
            
    #         # 2. Mask Invalid
    #         invalid_mask = ~torch.isfinite(scores_view)
    #         scores_view[invalid_mask] = -1e10

    #         # 3. Top-K Selection
    #         if select_mode == "softmax_resample":
    #             s = scores_view / temperature
    #             s = s - torch.max(s, dim=1, keepdim=True).values
    #             w = torch.softmax(s, dim=1)
    #             if not final:
    #                 selected_indices = torch.multinomial(w, num_samples=beam_width, replacement=True)
    #             else:
    #                 selected_indices = torch.multinomial(w, num_samples=1, replacement=True)
    #         else: # top_k
    #             if not final:
    #                 selected_indices = torch.topk(scores_view, k=beam_width, dim=1, largest=True).indices
    #             else:
    #                 selected_indices = torch.topk(scores_view, k=1, dim=1, largest=True).indices

    #         # 4. Gather Survivors
    #         gather_idx_3d = selected_indices.unsqueeze(-1).expand(-1, -1, D)
    #         theta_next_survivors = torch.gather(theta_next, 1, gather_idx_3d)
    #         theta1_pred_survivors = torch.gather(theta1_pred, 1, gather_idx_3d)

    #         # 5. Expand (Branching)
    #         if not final:
    #             theta_next_expanded = theta_next_survivors.repeat_interleave(branch_factor, dim=1)
    #             theta1_pred_expanded = theta1_pred_survivors.repeat_interleave(branch_factor, dim=1)
    #             return theta_next_expanded, theta1_pred_expanded
    #         else:
    #             return theta_next_survivors, theta1_pred_survivors

    #     # ==================================================================
    #     # 4. 主循环
    #     # ==================================================================
        
    #     # Init: [B, P, D]
    #     theta_flat = self.sample_theta_0(input_batch_size * n_candidates_full)
    #     theta = theta_flat.view(input_batch_size, n_candidates_full, -1)
        
    #     ts = self.make_sampling_schedule(reverse=False)
    #     n_steps = len(ts) - 1
        
    #     prune_start = int(prune_window[0] * n_steps)
    #     prune_end = int(prune_window[1] * n_steps)

    #     last_theta1_pred = None

    #     t1_avg_dict = {}
    #     t1_std_dict = {}
    #     for k in range(n_steps):
    #         u = ts[k]
    #         t = ts[k + 1]
    #         if k + score_every_prune > prune_end:
    #             theta_next, theta_1_pred = _step_update(
    #                 theta, u, t, context_data, alpha=alpha_after
    #             )
    #         elif k < prune_start:
    #             theta_next, theta_1_pred = _step_update(
    #                 theta, u, t, context_data, alpha=alpha_before
    #             )
    #         else:
    #             theta_next, theta_1_pred = _step_update(
    #                 theta, u, t, context_data, alpha=alpha_on
    #             )
            
    #         last_theta1_pred = theta_1_pred

    #         # 1. Score
    #         if _score_fn != None:
    #             scores = _score_fn(theta_1_pred) # Returns [B*P]
    #         else:
    #             if k <= prune_end:
    #                 scores = torch.ones(input_batch_size*n_candidates_full, device = theta_1_pred.device)
    #             else:
    #                 scores = torch.ones(input_batch_size, device = theta_1_pred.device)
    #         print(f"DEBUG [Step {k} Scoring]: scores.shape={scores.shape}, theta_1_pred.shape={theta_1_pred.shape}")
    #         # Stats Printing
    #         if k <= prune_end:
    #             scores_view = scores.view(input_batch_size, n_candidates_full)
    #         else:
    #             scores_view = scores.view(input_batch_size, 1)

    #         batch_max = scores_view.max(dim=1).values
    #         valid_max = batch_max[torch.isfinite(batch_max)]
    #         t1_avg = valid_max.mean().item() if len(valid_max) > 0 else float('nan')
    #         t1_std = valid_max.std().item() if len(valid_max) > 0 else float('nan')
    #         t1_avg_dict[str(k)] = t1_avg
    #         t1_std_dict[str(k)] = t1_std
    #         print(f"[Step {k}] Score Stats: Top-1 Mean={t1_avg:.2f}")
    #         # --- C. Prune & Expand ---
    #         if prune_start <= k <= prune_end:

    #             if (k - prune_start) % score_every_prune == 0:
                    
    #                 # # 1. Score
    #                 # if _score_fn != None:
    #                 #     scores = _score_fn(theta_1_pred) # Returns [B*P]
    #                 # else:
    #                 #     scores = torch.ones(input_batch_size*n_candidates_full, device = theta_1_pred.device)
    #                 # print(f"DEBUG [Step {k} Scoring]: scores.shape={scores.shape}, theta_1_pred.shape={theta_1_pred.shape}")
    #                 # # Stats Printing
    #                 # scores_view = scores.view(input_batch_size, n_candidates_full)
    #                 # batch_max = scores_view.max(dim=1).values
    #                 # valid_max = batch_max[torch.isfinite(batch_max)]
    #                 # t1_avg = valid_max.mean().item() if len(valid_max) > 0 else float('nan')
    #                 # t1_avg_dict[str(k)] = t1_avg
    #                 # print(f"[Step {k}] Score Stats: Top-1 Mean={t1_avg:.2f}")

    #                 # 2. Select & Expand
    #                 theta_1_pred = theta_1_pred.view(input_batch_size, n_candidates_full, -1)
    #                 if k + score_every_prune <= prune_end:
    #                     theta_next, theta_1_pred_expanded = _select_and_expand(
    #                         theta_next, theta_1_pred, scores
    #                     )
    #                 else:
    #                     theta_next, theta_1_pred_expanded = _select_and_expand(
    #                         theta_next, theta_1_pred, scores, final=True
    #                     )
                    
    #                 last_theta1_pred = theta_1_pred_expanded

    #         # Update State
    #         theta = theta_next

    #     # ==================================================================
    #     # 5. Output
    #     # ==================================================================
        
    #     target_tensor = last_theta1_pred if output_mode == "theta1_pred" else theta
    #     return target_tensor, t1_avg_dict, t1_std_dict
        # scores = _score_fn(target_tensor) # Returns [B*P]
        
            
        # # 1. Reshape Scores [B, P]
        # # scores_flat comes from _score_fn which returns [B*P]
        # scores_view = scores.view(input_batch_size, n_candidates_full)
        
        # # 2. Mask Invalid
        # invalid_mask = ~torch.isfinite(scores_view)
        # scores_view[invalid_mask] = -1e10

        # # 3. Top-K Selection
        # if select_mode == "softmax_resample":
        #     s = scores_view / temperature
        #     s = s - torch.max(s, dim=1, keepdim=True).values
        #     w = torch.softmax(s, dim=1)
        #     selected_indices = torch.multinomial(w, num_samples=beam_width, replacement=True)
        # else: # top_k
        #     selected_indices = torch.topk(scores_view, k=1, dim=1, largest=True).indices

        # # 4. Gather Survivors
        # gather_idx_3d = selected_indices.unsqueeze(-1).expand(-1, -1, target_tensor.shape[-1])
        # theta_next_survivors = torch.gather(target_tensor.view(input_batch_size, n_candidates_full, -1), 1, gather_idx_3d).squeeze(1)
        # return theta_next_survivors, t1_avg_dict
    
    @torch.no_grad()
    def sample(self, *context_data, num_samples: int = None, _score_fn=None, config=None):
        """
        Main sampling method (SDE + SMC Guidance), samples ONLY.
        Maintains explicit [Batch, Particles, Dim] shape.

        FK/SMC tempering form (annealing incremental potentials):
            G_j ∝ exp(Δβ_j * r_j)
        with linear β from 0 -> 1 across all resampling events, so ∑_j Δβ_j = 1.
        """

        # ==================================================================
        # 1. 配置 & 维度检查
        # ==================================================================
        if len(context_data) > 0:
            input_batch_size = context_data[0].shape[0] * num_samples
        else:
            raise ValueError("Context data required for guided sampling.")

        if config is None:
            beam_width = 4
            branch_factor = 4
            alpha_before = 0.0
            alpha_on = 1.0
            alpha_after = 0.0
            prune_window = (0.1, 0.95)
            score_every_prune = 2
            select_mode = "softmax_resample"
            temperature = 1.0
            output_mode = "theta1_pred"
        else:
            beam_width = config["beam_width"]
            branch_factor = config["branch_factor"]
            alpha_before = config["alpha_before"]
            alpha_on = config["alpha_on"]
            alpha_after = config["alpha_after"]
            prune_start_fraction = config["prune_start_fraction"]
            prune_end_fraction = config["prune_end_fraction"]
            score_every_prune = config["score_every_prune"]
            select_mode = config["select_mode"]
            temperature = config["temperature"]
            output_mode = config["output_mode"]
            prune_window = (prune_start_fraction, prune_end_fraction)

        n_candidates_full = beam_width * branch_factor

        print(
            f"Starting Guided Sampling (Samples Only). "
            f"Batch: {input_batch_size}, Beam: {beam_width}, Branch: {branch_factor}"
        )

        def _step_update(theta, u, t, context_list, alpha=0.0):
            """
            Input:  theta [B, P, D]
            Output: theta_next [B, P, D], theta1_pred_flat [B*P, D]
            """
            B, P, D = theta.shape
            current_total = B * P

            theta_flat = theta.view(current_total, D)

            t_vec = t.to(theta.device, theta.dtype).view(1).expand(current_total)
            u_vec = u.to(theta.device, theta.dtype).view(1).expand(current_total)
            dt = t_vec - u_vec

            predicted_vf = self.evaluate_vectorfield(u_vec, theta_flat, *context_list)

            # Predict theta_1 (Flat): your proxy for x0 / endpoint
            theta_1_pred_flat = theta_flat + (1 - u_vec.unsqueeze(1)) * predicted_vf

            # SDE Update (Flat)
            safe_t = torch.clamp(u_vec, min=1e-5).unsqueeze(1)
            sigma_t = alpha * torch.sqrt((1 - safe_t) / (safe_t + 0.1))
            correction = (sigma_t**2 / (2 * (1 - safe_t))) * (safe_t * predicted_vf - theta_flat)
            drift_theta = (predicted_vf + correction) * dt.unsqueeze(1)
            diffusion_theta = sigma_t * torch.sqrt(dt.unsqueeze(1)) * torch.randn_like(theta_flat)
            theta_next_flat = theta_flat + drift_theta + diffusion_theta

            theta_next = theta_next_flat.view(B, P, D)
            return theta_next, theta_1_pred_flat

        def _select_and_expand(theta_next, theta1_pred, scores_flat, delta_beta, final=False):
            """
            FK/SMC tempering:
                log w = (Δβ * r) / temperature
                w ∝ exp(log w)
            Input:
                theta_next  [B, P, D]
                theta1_pred [B, P, D]
                scores_flat [B*P]   # r (e.g., log p(x0|c))
                delta_beta  float
            Output:
                if not final: [B, beam*branch, D]
                if final:     [B, 1, D]
            """
            B, P, D = theta_next.shape

            # [B, P]
            scores_view = scores_flat.view(B, P)

            # log weights from annealed incremental potential
            log_w = (delta_beta * scores_view) / temperature

            # mask invalid
            log_w = log_w.clone()
            invalid_mask = ~torch.isfinite(log_w)
            log_w[invalid_mask] = -1e10

            # selection
            if select_mode == "softmax_resample":
                s = log_w - torch.max(log_w, dim=1, keepdim=True).values
                w = torch.softmax(s, dim=1)
                if not final:
                    selected_indices = torch.multinomial(w, num_samples=beam_width, replacement=True)
                else:
                    selected_indices = torch.multinomial(w, num_samples=1, replacement=True)
            else:  # top_k
                if not final:
                    selected_indices = torch.topk(log_w, k=beam_width, dim=1, largest=True).indices
                else:
                    selected_indices = torch.topk(log_w, k=1, dim=1, largest=True).indices

            # gather survivors
            gather_idx_3d = selected_indices.unsqueeze(-1).expand(-1, -1, D)
            theta_next_survivors = torch.gather(theta_next, 1, gather_idx_3d)
            theta1_pred_survivors = torch.gather(theta1_pred, 1, gather_idx_3d)

            # expand (branching)
            if not final:
                theta_next_expanded = theta_next_survivors.repeat_interleave(branch_factor, dim=1)
                theta1_pred_expanded = theta1_pred_survivors.repeat_interleave(branch_factor, dim=1)
                return theta_next_expanded, theta1_pred_expanded
            else:
                return theta_next_survivors, theta1_pred_survivors

        # ==================================================================
        # 4. 主循环
        # ==================================================================

        # Init: [B, P, D]
        theta_flat = self.sample_theta_0(input_batch_size * n_candidates_full)
        theta = theta_flat.view(input_batch_size, n_candidates_full, -1)

        ts = self.make_sampling_schedule(reverse=False)
        n_steps = len(ts) - 1

        prune_start = int(prune_window[0] * n_steps)
        prune_end = int(prune_window[1] * n_steps)

        # ----------------------------------------------------------
        # Annealing (linear beta): precompute how many resampling events
        # ----------------------------------------------------------
        resample_steps = [
            kk for kk in range(prune_start, prune_end + 1)
            if (kk - prune_start) % score_every_prune == 0
        ]
        m_resamples = max(1, len(resample_steps))
        beta_prev = 0.0
        resample_count = 0
        # ----------------------------------------------------------

        last_theta1_pred = None
        t1_avg_dict = {}
        t1_std_dict = {}

        for k in range(n_steps):
            u = ts[k]
            t = ts[k + 1]

            if k + score_every_prune > prune_end:
                theta_next, theta_1_pred = _step_update(theta, u, t, context_data, alpha=alpha_after)
            elif k < prune_start:
                theta_next, theta_1_pred = _step_update(theta, u, t, context_data, alpha=alpha_before)
            else:
                theta_next, theta_1_pred = _step_update(theta, u, t, context_data, alpha=alpha_on)

            last_theta1_pred = theta_1_pred  # flat [B*P, D]

            # current particle count (can change after final selection)
            B = input_batch_size
            P = theta_next.shape[1]

            # 1. Score (reward r)
            if _score_fn is not None:
                scores = _score_fn(theta_1_pred)  # expected shape [B*P]
            else:
                scores = torch.ones(B * P, device=theta_1_pred.device)

            print(f"DEBUG [Step {k} Scoring]: scores.shape={scores.shape}, theta_1_pred.shape={theta_1_pred.shape}")

            # Stats (always reshape by current P)
            scores_view = scores.view(B, P)
            batch_max = scores_view.max(dim=1).values
            valid_max = batch_max[torch.isfinite(batch_max)]
            t1_avg = valid_max.mean().item() if len(valid_max) > 0 else float("nan")
            t1_std = valid_max.std().item() if len(valid_max) > 0 else float("nan")
            t1_avg_dict[str(k)] = t1_avg
            t1_std_dict[str(k)] = t1_std
            print(f"[Step {k}] Score Stats: Top-1 Mean={t1_avg:.2f}")

            # --- C. Prune & Expand (only in prune window, and on schedule) ---
            if prune_start <= k <= prune_end and (k - prune_start) % score_every_prune == 0:

                # linear annealing: beta goes 0 -> 1 across all resampling events
                resample_count += 1
                beta_curr = resample_count / m_resamples
                beta_curr = max(0.0, min(1.0, beta_curr))
                delta_beta = beta_curr - beta_prev
                beta_prev = beta_curr

                # reshape theta_1_pred to [B, P, D] for gather
                theta_1_pred_3d = theta_1_pred.view(B, P, -1)

                # final selection if this is the LAST resampling event (beta reaches 1)
                is_final_resample = (resample_count == m_resamples)

                theta_next, theta_1_pred_expanded = _select_and_expand(
                    theta_next, theta_1_pred_3d, scores, delta_beta, final=is_final_resample
                )

                last_theta1_pred = theta_1_pred_expanded  # [B, P', D] (P' either n_candidates_full or 1)

            # Update State
            theta = theta_next

        # ==================================================================
        # 5. Output
        # ==================================================================
        target_tensor = last_theta1_pred if output_mode == "theta1_pred" else theta
        return target_tensor, t1_avg_dict, t1_std_dict

    
    # @torch.no_grad()
    # def sample_and_log_prob(self, *context_data, num_samples: int = None, _score_fn=None, config=None):
    #     """
    #     Main method for SDE sampling with SMC guidance, returning both samples and log-probabilities.
        
    #     Logic mirrors 'sample':
    #       - Start with (beam_width * branch_factor) particles.
    #       - SDE Update for Theta; ODE Divergence integration for Log_Prob.
    #       - If scoring step:
    #           1. Score candidates (Physical Likelihood).
    #           2. Select best 'beam_width' particles.
    #           3. Gather survivors (Theta AND Log_Prob).
    #           4. Immediately expand back to 'beam_width * branch_factor'.
    #       - Else (not scoring):
    #           Keep all particles.
    #       - Final: Select best 1 based on final score.
    #     """
        
    #     # ==================================================================
    #     # 1. 配置 & 维度检查
    #     # ==================================================================
    #     if len(context_data) > 0:
    #         input_batch_size = context_data[0].shape[0] * num_samples
    #     else:
    #         raise ValueError("Context data required for guided sampling.")

    #     if config == None:
    #         beam_width = 4          
    #         branch_factor = 4     
    #         alpha_before = 0.0
    #         alpha_on = 0.7
    #         alpha_after = 0.0     
    #         prune_window = (0.25, 0.85)   
    #         score_every_prune = 4       
    #         select_mode = "softmax_resample" 
    #         temperature = 1.0           
    #         output_mode = "theta1_pred" 
    #     else:
    #         beam_width = config["beam_width"]
    #         branch_factor = config["branch_factor"]
    #         alpha_before = config["alpha_before"]
    #         alpha_on = config["alpha_on"]
    #         alpha_after = config["alpha_after"]
    #         prune_start_fraction = config["prune_start_fraction"]
    #         prune_end_fraction = config["prune_end_fraction"]
    #         score_every_prune = config["score_every_prune"]
    #         select_mode = config["select_mode"]
    #         temperature = config["temperature"]
    #         output_mode = config["output_mode"]
    #         prune_window = (prune_start_fraction, prune_end_fraction)
        
    #     n_candidates_full = beam_width * branch_factor 

    #     print(f"Starting Guided Sampling (Samples Only). Batch: {input_batch_size}, Beam: {beam_width}, Branch: {branch_factor}")

    #     def _step_update(theta, u, t, context_list, alpha = 0.0):
    #         """
    #         Input:  theta [B, P, D]
    #         Output: theta [B, P, D], theta1_pred [B, P, D]
    #         """
    #         B, P, D = theta.shape
    #         current_total = B * P
            
    #         # 1. Flatten for Network
    #         theta_flat = theta.view(current_total, D)
            
    #         t_vec = t.to(theta.device, theta.dtype).view(1).expand(current_total)
    #         u_vec = u.to(theta.device, theta.dtype).view(1).expand(current_total)
    #         dt = t_vec - u_vec 

    #         # 2. Network Calculation (No Divergence needed)
    #         # context_list is already expanded to [B*P, ...]
    #         # evaluate_vectorfield returns [B*P, D]
    #         predicted_vf = self.evaluate_vectorfield(u_vec, theta_flat, *context_list)
            
    #         # 3. Predict theta_1 (Flat)
    #         theta_1_pred_flat = theta_flat + (1 - u_vec.unsqueeze(1)) * predicted_vf

    #         # 4. SDE Update (Flat)
    #         safe_t = torch.clamp(u_vec, min=1e-5).unsqueeze(1)
    #         sigma_t = alpha * torch.sqrt((1 - safe_t) / (safe_t + 0.1))
    #         correction = (sigma_t**2 / (2 * (1 - safe_t))) * (safe_t * predicted_vf - theta_flat)
    #         drift_theta = (predicted_vf + correction) * dt.unsqueeze(1)
    #         diffusion_theta = sigma_t * torch.sqrt(dt.unsqueeze(1)) * torch.randn_like(theta_flat)
    #         theta_next_flat = theta_flat + drift_theta + diffusion_theta
            
    #         mean = theta_flat + drift_theta

    #         # var: [N, 1] -> broadcast to [N, D]
    #         var = (sigma_t ** 2) * dt.unsqueeze(1)          # [N,1]
    #         var = torch.clamp(var, min=1e-12)              # numerical safety
    #         log_var = torch.log(var)

    #         # log_prob per dimension then sum over D -> [N]
    #         quad = (theta_next_flat - mean) ** 2 / var
    #         log_prob_flat = -0.5 * (quad + log_var + math.log(2 * math.pi))
    #         # 5. Unflatten -> [B, P, D]
    #         theta_next = theta_next_flat.view(B, P, D)
    #         log_prob = log_prob_flat.view(B, P, D)
    #         # theta_1_pred = theta_1_pred_flat.view(B, P, D)

    #         return theta_next, theta_1_pred_flat, log_prob

    #     def _select_and_expand(theta_next, theta1_pred, scores_flat, log_prob, final = False):
    #         """
    #         Input: [Batch, Current_Particles, D]
    #         Output: [Batch, Beam * Branch, D]
    #         """
    #         B, P, D = theta_next.shape
            
    #         # 1. Reshape Scores [B, P]
    #         # scores_flat comes from _score_fn which returns [B*P]
    #         scores_view = scores_flat.view(B, P)
            
    #         # 2. Mask Invalid
    #         invalid_mask = ~torch.isfinite(scores_view)
    #         scores_view[invalid_mask] = -1e10

    #         # 3. Top-K Selection
    #         if select_mode == "softmax_resample":
    #             s = scores_view / temperature
    #             s = s - torch.max(s, dim=1, keepdim=True).values
    #             w = torch.softmax(s, dim=1)
    #             if not final:
    #                 selected_indices = torch.multinomial(w, num_samples=beam_width, replacement=True)
    #             else:
    #                 selected_indices = torch.multinomial(w, num_samples=1, replacement=True)
    #         else: # top_k
    #             if not final:
    #                 selected_indices = torch.topk(scores_view, k=beam_width, dim=1, largest=True).indices
    #             else:
    #                 selected_indices = torch.topk(scores_view, k=1, dim=1, largest=True).indices

    #         # 4. Gather Survivors
    #         gather_idx_3d = selected_indices.unsqueeze(-1).expand(-1, -1, D)
    #         theta_next_survivors = torch.gather(theta_next, 1, gather_idx_3d)
    #         theta1_pred_survivors = torch.gather(theta1_pred, 1, gather_idx_3d)
    #         log_prob_survivors = torch.gather(log_prob, 1, gather_idx_3d)
    #         # 5. Expand (Branching)
    #         if not final:
    #             theta_next_expanded = theta_next_survivors.repeat_interleave(branch_factor, dim=1)
    #             theta1_pred_expanded = theta1_pred_survivors.repeat_interleave(branch_factor, dim=1)
    #             log_prob_expanded = log_prob_survivors.repeat_interleave(branch_factor, dim=1)
    #             return theta_next_expanded, theta1_pred_expanded, log_prob_expanded
    #         else:
    #             return theta_next_survivors, theta1_pred_survivors, log_prob_survivors

    #     # ==================================================================
    #     # 4. 主循环
    #     # ==================================================================
        
    #     # Init: [B, P, D]
    #     theta_flat = self.sample_theta_0(input_batch_size * n_candidates_full)
    #     theta = theta_flat.view(input_batch_size, n_candidates_full, -1)
    #     log_prob_expanded = torch.zeros_like(theta)

    #     ts = self.make_sampling_schedule(reverse=False)
    #     n_steps = len(ts) - 1
        
    #     prune_start = int(prune_window[0] * n_steps)
    #     prune_end = int(prune_window[1] * n_steps)

    #     last_theta1_pred = None

    #     t1_avg_dict = {}
    #     t1_std_dict = {}
    #     for k in range(n_steps):
    #         u = ts[k]
    #         t = ts[k + 1]
    #         if k + score_every_prune >= prune_end:
    #             theta_next, theta_1_pred, log_prob = _step_update(
    #                 theta, u, t, context_data, alpha=alpha_after
    #             )
    #         elif k < prune_start:
    #             theta_next, theta_1_pred, log_prob = _step_update(
    #                 theta, u, t, context_data, alpha=alpha_before
    #             )
    #         else:
    #             theta_next, theta_1_pred, log_prob = _step_update(
    #                 theta, u, t, context_data, alpha=alpha_on
    #             )
            
    #         log_prob_expanded = log_prob_expanded + log_prob
    #         last_theta1_pred = theta_1_pred

    #         # 1. Score
    #         if _score_fn != None:
    #             scores = _score_fn(theta_1_pred) # Returns [B*P]
    #         else:
    #             if k <= prune_end:
    #                 scores = torch.ones(input_batch_size*n_candidates_full, device = theta_1_pred.device)
    #             else:
    #                 scores = torch.ones(input_batch_size, device = theta_1_pred.device)
    #         print(f"DEBUG [Step {k} Scoring]: scores.shape={scores.shape}, theta_1_pred.shape={theta_1_pred.shape}")
    #         # Stats Printing
    #         if k <= prune_end:
    #             scores_view = scores.view(input_batch_size, n_candidates_full)
    #         else:
    #             scores_view = scores.view(input_batch_size, 1)

    #         batch_max = scores_view.max(dim=1).values
    #         valid_max = batch_max[torch.isfinite(batch_max)]
    #         t1_avg = valid_max.mean().item() if len(valid_max) > 0 else float('nan')
    #         t1_std = valid_max.std().item() if len(valid_max) > 0 else float('nan')
    #         t1_avg_dict[str(k)] = t1_avg
    #         t1_std_dict[str(k)] = t1_std
    #         print(f"[Step {k}] Score Stats: Top-1 Mean={t1_avg:.2f}")
    #         # --- C. Prune & Expand ---
    #         if prune_start <= k <= prune_end:

    #             if (k - prune_start) % score_every_prune == 0:
                    
    #                 # # 1. Score
    #                 # if _score_fn != None:
    #                 #     scores = _score_fn(theta_1_pred) # Returns [B*P]
    #                 # else:
    #                 #     scores = torch.ones(input_batch_size*n_candidates_full, device = theta_1_pred.device)
    #                 # print(f"DEBUG [Step {k} Scoring]: scores.shape={scores.shape}, theta_1_pred.shape={theta_1_pred.shape}")
    #                 # # Stats Printing
    #                 # scores_view = scores.view(input_batch_size, n_candidates_full)
    #                 # batch_max = scores_view.max(dim=1).values
    #                 # valid_max = batch_max[torch.isfinite(batch_max)]
    #                 # t1_avg = valid_max.mean().item() if len(valid_max) > 0 else float('nan')
    #                 # t1_avg_dict[str(k)] = t1_avg
    #                 # print(f"[Step {k}] Score Stats: Top-1 Mean={t1_avg:.2f}")

    #                 # 2. Select & Expand
    #                 theta_1_pred = theta_1_pred.view(input_batch_size, n_candidates_full, -1)
    #                 if k + score_every_prune <= prune_end:
    #                     theta_next, theta_1_pred_expanded, log_prob_expanded = _select_and_expand(
    #                         theta_next, theta_1_pred, scores, log_prob_expanded
    #                     )
    #                 else:
    #                     theta_next, theta_1_pred_expanded, log_prob_expanded = _select_and_expand(
    #                         theta_next, theta_1_pred, scores, log_prob_expanded, final=True
    #                     )
                    
    #                 last_theta1_pred = theta_1_pred_expanded

    #         # Update State
    #         theta = theta_next

    #     # ==================================================================
    #     # 5. Output
    #     # ==================================================================
        
    #     target_tensor = last_theta1_pred if output_mode == "theta1_pred" else theta
    #     return target_tensor, t1_avg_dict, t1_std_dict, log_prob_expanded
    @torch.no_grad()
    def sample_and_log_prob(self, *context_data, num_samples: int = None, _score_fn=None, config=None):
        """
        Main method for SDE sampling with SMC guidance, returning both samples and log-probabilities.

        Uses FK/SMC tempering (annealing incremental potentials):
            G_j ∝ exp(Δβ_j * r_j)
        with linear β from 0 -> 1 across all resampling events, so ∑_j Δβ_j = 1.

        Keeps the ORIGINAL per-step Gaussian transition log-prob calculation unchanged.
        """

        # ==================================================================
        # 1. 配置 & 维度检查
        # ==================================================================
        if len(context_data) > 0:
            input_batch_size = context_data[0].shape[0] * num_samples 
        else:
            raise ValueError("Context data required for guided sampling.")

        if config is None:
            beam_width = 8
            branch_factor = 1
            alpha_before = 0.0
            alpha_on = 0.3
            alpha_after = 0.0
            prune_window = (0.1, 1.0)
            score_every_prune = 5
            select_mode = "softmax_resample"
            temperature = 1.0
            output_mode = "theta1_pred"
        else:
            beam_width = config["beam_width"]
            branch_factor = config["branch_factor"]
            alpha_before = config["alpha_before"]
            alpha_on = config["alpha_on"]
            alpha_after = config["alpha_after"]
            prune_start_fraction = config["prune_start_fraction"]
            prune_end_fraction = config["prune_end_fraction"]
            score_every_prune = config["score_every_prune"]
            select_mode = config["select_mode"]
            temperature = config["temperature"]
            output_mode = config["output_mode"]
            prune_window = (prune_start_fraction, prune_end_fraction)

        n_candidates_full = beam_width * branch_factor

        # input_batch_size = input_batch_size  // beam_width
        print(
            f"Starting Guided Sampling (Samples+LogProb). "
            f"Batch: {input_batch_size}, Beam: {beam_width}, Branch: {branch_factor}"
        )

        def _step_update(theta, u, t, context_list, alpha=0.0):
            """
            Input:  theta [B, P, D]
            Output: theta_next [B, P, D], theta1_pred_flat [B*P, D], log_prob [B, P, D]
            """
            B, P, D = theta.shape
            current_total = B * P

            # 1. Flatten for Network
            theta_flat = theta.view(current_total, D)

            t_vec = t.to(theta.device, theta.dtype).view(1).expand(current_total)
            u_vec = u.to(theta.device, theta.dtype).view(1).expand(current_total)
            dt = t_vec - u_vec

            # 2. Network Calculation
            predicted_vf = self.evaluate_vectorfield(u_vec, theta_flat, *context_list)

            # 3. Predict theta_1 (Flat)
            theta_1_pred_flat = theta_flat + (1 - u_vec.unsqueeze(1)) * predicted_vf

            # 4. SDE Update (Flat)
            safe_t = torch.clamp(u_vec, min=1e-2).unsqueeze(1)
            sigma_t = alpha * torch.sqrt((1 - safe_t) / (safe_t))
            correction = (sigma_t**2 / (2 * (1 - safe_t))) * (safe_t * predicted_vf - theta_flat)
            drift_theta = (predicted_vf + correction) * dt.unsqueeze(1)
            diffusion_theta = sigma_t * torch.sqrt(dt.unsqueeze(1)) * torch.randn_like(theta_flat)
            theta_next_flat = theta_flat + drift_theta + diffusion_theta

            # ===== ORIGINAL log-prob computation (unchanged) =====
            mean = theta_flat + drift_theta
            var = (sigma_t ** 2) * dt.unsqueeze(1)          # [N,1]
            var = torch.clamp(var, min=1e-12)
            log_var = torch.log(var)
            quad = (theta_next_flat - mean) ** 2 / var
            log_prob_flat = -0.5 * (quad + log_var + math.log(2 * math.pi))
            # ================================================

            theta_next = theta_next_flat.view(B, P, D)
            log_prob = log_prob_flat.view(B, P, D)
            return theta_next, theta_1_pred_flat, log_prob

        def _select_and_expand(theta_next, theta1_pred, scores_flat, log_prob_accum, delta_beta, final=False):
            """
            FK/SMC tempering selection with annealed incremental potential:
                log w = (Δβ * r) / temperature

            Input:
                theta_next     [B, P, D]
                theta1_pred    [B, P, D]
                scores_flat    [B*P]      (r)
                log_prob_accum [B, P, D]  (accumulated log prob up to current step)
                delta_beta     float
            Output:
                if not final: [B, beam*branch, D]
                if final:     [B, 1, D]
            """
            B, P, D = theta_next.shape

            # 1) reshape reward to [B, P]
            scores_view = scores_flat.view(B, P)

            # 2) annealed incremental log weights
            log_w = (delta_beta * scores_view) / temperature

            # 3) mask invalid
            log_w = log_w.clone()
            invalid_mask = ~torch.isfinite(log_w)
            log_w[invalid_mask] = -1e10

            # 4) selection
            if select_mode == "softmax_resample":
                s = log_w - torch.max(log_w, dim=1, keepdim=True).values
                w = torch.softmax(s, dim=1)
                if not final:
                    selected_indices = torch.multinomial(w, num_samples=beam_width, replacement=True)
                else:
                    selected_indices = torch.multinomial(w, num_samples=1, replacement=True)
            else:  # top_k
                if not final:
                    selected_indices = torch.topk(log_w, k=beam_width, dim=1, largest=True).indices
                else:
                    selected_indices = torch.topk(log_w, k=1, dim=1, largest=True).indices

            # 5) gather survivors (theta / theta1_pred / log_prob)
            gather_idx_3d = selected_indices.unsqueeze(-1).expand(-1, -1, D)
            theta_next_survivors = torch.gather(theta_next, 1, gather_idx_3d)
            theta1_pred_survivors = torch.gather(theta1_pred, 1, gather_idx_3d)
            log_prob_survivors = torch.gather(log_prob_accum, 1, gather_idx_3d)

            # 6) expand (branching)
            if not final:
                theta_next_expanded = theta_next_survivors.repeat_interleave(branch_factor, dim=1)
                theta1_pred_expanded = theta1_pred_survivors.repeat_interleave(branch_factor, dim=1)
                log_prob_expanded = log_prob_survivors.repeat_interleave(branch_factor, dim=1)
                return theta_next_expanded, theta1_pred_expanded, log_prob_expanded
            else:
                return theta_next_survivors, theta1_pred_survivors, log_prob_survivors

        # ==================================================================
        # 4. 主循环
        # ==================================================================

        # Init: [B, P, D]
        theta_flat = self.sample_theta_0(input_batch_size * 1)
        theta = theta_flat.view(input_batch_size, 1, -1)
        log_prob_accum = torch.zeros_like(theta)
        theta = theta.repeat_interleave(beam_width * branch_factor, dim=1)
        log_prob_accum = log_prob_accum.repeat_interleave(beam_width * branch_factor, dim=1)

        ts = self.make_sampling_schedule(reverse=False)
        n_steps = len(ts) - 1

        prune_start = int(prune_window[0] * n_steps)
        prune_end = int(prune_window[1] * n_steps)
        print("Annealing (linear beta): precompute number of resampling events")
        # ----------------------------------------------------------
        # Annealing (linear beta): precompute number of resampling events
        # ----------------------------------------------------------
        resample_steps = [
            kk for kk in range(prune_start, prune_end + 1)
            if (kk - prune_start) % score_every_prune == 0
        ]
        m_resamples = max(1, len(resample_steps))
        beta_prev = 0.0
        resample_count = 0
        # ----------------------------------------------------------

        t1_avg_dict = {}
        t1_std_dict = {}

        for k in range(n_steps):
            u = ts[k]
            t = ts[k + 1]

            if k + score_every_prune > prune_end:
                theta_next, theta_1_pred, log_prob = _step_update(theta, u, t, context_data, alpha=alpha_after)
            elif k < prune_start:
                theta_next, theta_1_pred, log_prob = _step_update(theta, u, t, context_data, alpha=alpha_before)
            else:
                theta_next, theta_1_pred, log_prob = _step_update(theta, u, t, context_data, alpha=alpha_on)

            # accumulate log prob (shape-consistent)
            log_prob_accum = log_prob_accum + log_prob

            # if k + 1 == prune_start:
            #     theta_next = theta_next.repeat_interleave(beam_width * branch_factor, dim=1)
            #     theta_1_pred = theta_1_pred.repeat_interleave(beam_width * branch_factor, dim=0)
            #     log_prob = log_prob.repeat_interleave(beam_width * branch_factor, dim=1)
            # current P can change after final resample
            B = input_batch_size
            P = theta_next.shape[1]

            # --- C. Prune & Expand (annealed incremental potential) ---
            if prune_start <= k <= prune_end and (k - prune_start) % score_every_prune == 0:
                # 1) Score (reward r) on theta_1_pred (flat)
                if _score_fn is not None:
                    scores = _score_fn(theta_1_pred)  # expected [B*P]
                else:
                    scores = torch.ones(B * P, device=theta_1_pred.device)

                print(f"DEBUG [Step {k} Scoring]: scores.shape={scores.shape}, theta_1_pred.shape={theta_1_pred.shape}")

                # Stats printing (use current P)
                scores_view = scores.view(B, P)
                batch_max = scores_view.max(dim=1).values
                valid_max = batch_max[torch.isfinite(batch_max)]
                t1_avg = valid_max.mean().item() if len(valid_max) > 0 else float("nan")
                t1_std = valid_max.std().item() if len(valid_max) > 0 else float("nan")
                t1_avg_dict[str(k)] = t1_avg
                t1_std_dict[str(k)] = t1_std
                print(f"[Step {k}] Score Stats: Top-1 Mean={t1_avg:.2f}")
                # update linear beta
                resample_count += 1
                beta_curr = resample_count / m_resamples
                beta_curr = max(0.0, min(1.0, beta_curr))
                delta_beta = beta_curr - beta_prev
                beta_prev = beta_curr

                # reshape theta_1_pred to [B, P, D] for gather
                theta_1_pred_3d = theta_1_pred.view(B, P, -1)

                # final selection only at the last resampling event
                is_final_resample = (resample_count == m_resamples)

                theta_next, theta_1_pred_out, log_prob_accum = _select_and_expand(
                    theta_next,
                    theta_1_pred_3d,
                    scores,
                    log_prob_accum,
                    delta_beta,
                    final=is_final_resample,
                )


            # Update State
            theta = theta_next

        # ==================================================================
        # 5. Output
        # ==================================================================
        target_tensor = theta.view(B*P, -1)
        return target_tensor, t1_avg_dict, t1_std_dict, log_prob_accum

    
def ot_conditional_flow(x_0, x_1, t, sigma_min):
    return (1 - (1 - sigma_min) * t)[:, None] * x_0 + t[:, None] * x_1
