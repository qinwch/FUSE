import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# -----------------------------------------------------------------------------
# Basic Components
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization for improved numerical stability.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.g * self.scale

class TimeStepEmbedding(nn.Module):
    """
    Embeds scalar time 't' into a high-dimensional vector using sinusoidal encoding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        # t shape: (Batch, 1) or (Batch,)
        if t.ndim == 2:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)

class FeedForward(nn.Module):
    """
    Standard MLP inside the Transformer block.
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class JointAttention(nn.Module):
    """
    Multi-modal attention that allows Context and Theta tokens to interact.
    """
    def __init__(self, dim, heads=4, dim_head=16):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.to_qkv = nn.ModuleList([
            nn.Linear(dim, inner_dim * 3, bias=False),
            nn.Linear(dim, inner_dim * 3, bias=False)
        ])
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, dim),
            nn.Linear(inner_dim, dim)
        ])

    def forward(self, ctx, theta):
        # ctx/theta shape: (B, N, D)
        qkv_ctx = self.to_qkv[0](ctx).chunk(3, dim=-1)
        qkv_theta = self.to_qkv[1](theta).chunk(3, dim=-1)

        def reshape_qkv(t):
            return rearrange(t, 'b n (h d) -> b h n d', h=self.heads)
        
        qc, kc, vc = map(reshape_qkv, qkv_ctx)
        qt, kt, vt = map(reshape_qkv, qkv_theta)

        # Joint processing
        q = torch.cat((qc, qt), dim=2)
        k = torch.cat((kc, kt), dim=2)
        v = torch.cat((vc, vt), dim=2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)

        # Separate back to context and theta
        out_ctx = out[:, :, :ctx.shape[1], :]
        out_theta = out[:, :, ctx.shape[1]:, :]

        out_ctx = rearrange(out_ctx, 'b h n d -> b n (h d)')
        out_theta = rearrange(out_theta, 'b h n d -> b n (h d)')

        return self.to_out[0](out_ctx), self.to_out[1](out_theta)

# -----------------------------------------------------------------------------
# MMDiT Block
# -----------------------------------------------------------------------------

class MMDiTBlock(nn.Module):
    """
    MM-DiT block with Adaptive LayerNorm (AdaLN-Zero).
    """
    def __init__(self, dim, heads, dim_head, dim_cond):
        super().__init__()
        self.attn = JointAttention(dim, heads, dim_head)
        self.ff_ctx = FeedForward(dim)
        self.ff_theta = FeedForward(dim)

        self.norm_ctx = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_theta = nn.LayerNorm(dim, elementwise_affine=False)

        # 12 params: (scale, shift, gate) * 2 streams * 2 sub-blocks (Attn/FF)
        self.to_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, dim * 12)
        )
        
        # Initialize the projection to zero to ensure identity at start
        nn.init.zeros_(self.to_cond[1].weight)
        nn.init.zeros_(self.to_cond[1].bias)

    def forward(self, ctx, theta, cond):
        modulation = self.to_cond(cond).chunk(12, dim=-1)
        (tc_as, tc_ab, tc_ag, tc_fs, tc_fb, tc_fg, 
         tt_as, tt_ab, tt_ag, tt_fs, tt_fb, tt_fg) = [m.unsqueeze(1) for m in modulation]

        # Attention block
        res_ctx, res_theta = ctx, theta
        ctx_n = self.norm_ctx(ctx) * (1 + tc_as) + tc_ab
        theta_n = self.norm_theta(theta) * (1 + tt_as) + tt_ab
        attn_ctx, attn_theta = self.attn(ctx_n, theta_n)
        ctx = res_ctx + tc_ag * attn_ctx
        theta = res_theta + tt_ag * attn_theta

        # Feed-forward block
        res_ctx, res_theta = ctx, theta
        ctx_n = self.norm_ctx(ctx) * (1 + tc_fs) + tc_fb
        theta_n = self.norm_theta(theta) * (1 + tt_fs) + tt_fb
        ctx = res_ctx + tc_fg * self.ff_ctx(ctx_n)
        theta = res_theta + tt_fg * self.ff_theta(theta_n)
        
        return ctx, theta

# -----------------------------------------------------------------------------
# Dingo Model Adapter
# -----------------------------------------------------------------------------

class DingoMMDiT(nn.Module):
    """
    Main MM-DiT model adapted for Dingo GW Posterior Estimation.
    """
    def __init__(
        self,
        input_dim: int,       # Concatenated dim [Context (dim=C) | t (dim=1) | Theta (dim=P)]
        output_dim: int,      # Posterior dimension (same as P)
        context_dim: int,     # Dimensionality of the context part
        hidden_dim: int = 64, # Your experiment showed 64 is better than 128
        depth: int = 4,
        heads: int = 4,
        num_tokens: int = 4   # Number of tokens to split the small vectors into
    ):
        super().__init__()
        self.context_dim = context_dim
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        
        # dingo input x: [Context | t | Theta]
        # So theta part length is: input_dim - context_dim - 1 (for t)
        self.theta_params_dim = input_dim - context_dim - 1
        
        # 1. Time Embedding (Flow steering signal)
        self.time_emb_net = TimeStepEmbedding(hidden_dim)
        
        # 2. Input Projections (Vector to Sequence)
        self.ctx_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * num_tokens),
            nn.SiLU(),
            nn.Linear(hidden_dim * num_tokens, hidden_dim * num_tokens)
        )
        self.theta_proj = nn.Sequential(
            nn.Linear(self.theta_params_dim, hidden_dim * num_tokens),
            nn.SiLU(),
            nn.Linear(hidden_dim * num_tokens, hidden_dim * num_tokens)
        )

        # 3. Learnable Positional Embeddings
        # Helps the Transformer know which token represents which physical parameter
        self.ctx_pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.theta_pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        
        # 4. Global Condition Projector (Injects strain information into AdaLN)
        self.cond_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 5. MM-DiT Transformer Blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden_dim, heads, dim_head=hidden_dim // heads, dim_cond=hidden_dim)
            for _ in range(depth)
        ])

        # 6. Final Output Head
        self.final_norm = RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim * num_tokens, output_dim)
        
        # Initialize output to zero for training stability
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, theta, t, glu_context=None):
        """
        [Theta | t | Context]
        """
        # 1. Unpack Dingo's concatenated input
        # Context is the encoded strain data
        ctx_vec = glu_context
        
        theta_params = theta # Remaining columns are physical parameters

        # 2. Prepare Condition signal for AdaLN (Time + Context)
        t_emb = self.time_emb_net(t)
        c_emb = self.cond_proj(ctx_vec)
        cond = t_emb + c_emb 

        # 3. Project vectors into tokens and add position information
        ctx = rearrange(self.ctx_proj(ctx_vec), 'b (n d) -> b n d', n=self.num_tokens)
        theta = rearrange(self.theta_proj(theta_params), 'b (n d) -> b n d', n=self.num_tokens)
        
        ctx = ctx + self.ctx_pos_embed
        theta = theta + self.theta_pos_embed

        # 4. Pass through MM-DiT Blocks
        for block in self.blocks:
            ctx, theta = block(ctx, theta, cond)

        # 5. Final Output (Sequence to Vector)
        theta = self.final_norm(theta)
        theta = rearrange(theta, 'b n d -> b (n d)')
        
        return self.head(theta)
    
class DingoMMDiTV2(nn.Module):
    """
    Main MM-DiT model adapted for Dingo GW Posterior Estimation.
    """
    def __init__(
        self,
        theta_dim: int,       # Theta dimension
        context_dim: int,     # Dimensionality of the context part
        hidden_dim: int = 64, # Your experiment showed 64 is better than 128
        depth: int = 4,
        heads: int = 4,
        context_num_tokens: int = 4,   # Number of tokens to split the context vector into
        theta_num_tokens: int = 4,      # Number of tokens to split the theta vector into
        is_individual: bool = False,
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.context_dim = context_dim
        self.context_num_tokens = context_num_tokens
        self.theta_num_tokens = theta_num_tokens
        self.hidden_dim = hidden_dim
        self.is_individual = is_individual
        # dingo input x: [Context | t | Theta]
        # So theta part length is: input_dim - context_dim - 1 (for t)
        
        # 1. Time Embedding (Flow steering signal)
        self.time_emb_net = TimeStepEmbedding(hidden_dim)
        
        # 2. Input Projections (Vector to Sequence)
        self.ctx_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * context_num_tokens),
            nn.SiLU(),
            nn.Linear(hidden_dim * context_num_tokens, hidden_dim * context_num_tokens)
        )
        if not is_individual:
            self.theta_proj = nn.Sequential(
                nn.Linear(self.theta_dim, hidden_dim * theta_num_tokens),
                nn.SiLU(),
                nn.Linear(hidden_dim * theta_num_tokens, hidden_dim * theta_num_tokens)
            )
        else:
            self.theta_proj = nn.Sequential(
                nn.Linear(1, hidden_dim * theta_num_tokens),
                nn.SiLU(),
                nn.Linear(hidden_dim * theta_num_tokens, hidden_dim * theta_num_tokens)
            )

        # 3. Learnable Positional Embeddings
        self.ctx_pos_embed = nn.Embedding(context_num_tokens, hidden_dim)
        if not is_individual:
            self.theta_pos_embed = nn.Embedding(theta_num_tokens, hidden_dim)
        else:
            self.theta_pos_embed = nn.Embedding(theta_num_tokens*theta_dim, hidden_dim)
        
        # 4. Global Condition Projector (Injects strain information into AdaLN)
        self.cond_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 5. MM-DiT Transformer Blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden_dim, heads, dim_head=hidden_dim // heads, dim_cond=hidden_dim)
            for _ in range(depth)
        ])

        # 6. Final Output Head
        self.final_norm = RMSNorm(hidden_dim)
        if not is_individual:
            self.head = nn.Linear(hidden_dim * theta_num_tokens, theta_dim)
        else:
            self.head = nn.Linear(hidden_dim * theta_num_tokens, 1)
        
    def forward(self, theta, t, glu_context=None):
        """
        [Theta | t | Context]
        """
        # 1. Unpack Dingo's concatenated input
        # Context is the encoded strain data
        ctx_vec = glu_context.to(theta.device)
        
        theta_params = theta # Remaining columns are physical parameters

        # 2. Prepare Condition signal for AdaLN (Time + Context)
        t_emb = self.time_emb_net(t)
        c_emb = self.cond_proj(ctx_vec)
        cond = t_emb + c_emb 

        # 3. Project vectors into tokens and add position information
        ctx = rearrange(self.ctx_proj(ctx_vec), 'b (n d) -> b n d', n=self.context_num_tokens)
        if not self.is_individual:
            theta_params = rearrange(self.theta_proj(theta), 'b (n d) -> b n d', n=self.theta_num_tokens)
        else:
            theta_params = theta[..., None]
            theta_params = rearrange(self.theta_proj(theta_params), 'b t (n d) -> b (t n) d', n=self.theta_num_tokens)
        
        ctx = ctx + self.ctx_pos_embed(torch.arange(self.context_num_tokens, device=ctx.device))[None, :, :]
        theta_params = theta_params + self.theta_pos_embed(torch.arange(theta_params.shape[1], device=theta_params.device))[None, :, :]

        # 4. Pass through MM-DiT Blocks
        for block in self.blocks:
            ctx, theta_params = block(ctx, theta_params, cond)

        # 5. Final Output (Sequence to Vector)
        theta_params = self.final_norm(theta_params)
        if not self.is_individual:
            theta_params = rearrange(theta_params, 'b n d -> b (n d)')
            return self.head(theta_params)
        else:
            theta_params = rearrange(theta_params, 'b (t n) d -> b t (n d)', n=self.theta_num_tokens)
            out = self.head(theta_params)
            return out.squeeze(-1)
        