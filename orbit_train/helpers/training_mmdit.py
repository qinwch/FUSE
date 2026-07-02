import wandb
from tqdm import tqdm
from lampe.utils import GDStep
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn as nn
from itertools import islice
from lampe.inference import  FMPELoss
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from pathlib import Path

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
        

class MMDiTFMPE(nn.Module):
    r"""
    A Flow Matching Posterior Estimator powered by MM-DiT.
    Replaces the standard MLP with DingoMMDiTV2 for better context-parameter interaction.
    """

    def __init__(
        self,
        theta_dim: int,
        context_dim: int,     # 对应原来的 x_dim
        hidden_dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        context_num_tokens: int = 4,
        theta_num_tokens: int = 4,
        is_individual: bool = False,
    ):
        super().__init__()
        
        # 初始化 MM-DiT 核心网络
        self.net = DingoMMDiTV2(
            theta_dim=theta_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            heads=heads,
            context_num_tokens=context_num_tokens,
            theta_num_tokens=theta_num_tokens,
            is_individual=is_individual
        )

        # 用于 flow 方法中的基分布参数
        self.register_buffer("zeros", torch.zeros(theta_dim))
        self.register_buffer("ones", torch.ones(theta_dim))

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: Parameters (Batch, D)
            x: Observation / Context (Batch, L)
            t: Time (Batch,) or (Batch, 1)

        Returns:
            Vector field v (Batch, D)
        """
        # 1. 维度处理
        # DingoMMDiTV2 内部会自动处理 TimeStepEmbedding，所以这里不需要像原版 FMPE 那样手动做 cos/sin 编码
        # 我们只需要确保 t 是正确的形状 (Batch,) 或 (Batch, 1)
        if t.ndim == 1:
            t = t.unsqueeze(-1) # (B,) -> (B, 1)

        # 2. 调用 MM-DiT
        # 注意：DingoMMDiTV2 的参数名为 (theta, t, glu_context)
        # 这里 x 对应 glu_context (观测数据)
        v = self.net(theta=theta, t=t, glu_context=x)

        return v

    def flow(self, x: Tensor) -> Distribution:
        r"""
        Constructs the normalizing flow p(theta | x) for inference.
        """
        # 定义 ODE 求解所需的函数形式 f(t, theta)
        # 注意 zuko 的 FreeFormJacobianTransform 期望 f(t, theta)
        # 但我们需要把 x (context) 传进去
        def vector_field(t, theta):
            # t 可能是标量，需要扩展到 batch size
            if t.numel() == 1:
                t = t.expand(theta.shape[0], 1)
            # x 需要扩展以匹配 theta 的 batch size (如果在采样时 theta 是多个粒子)
            x_expanded = x.expand(theta.shape[0], -1)
            return self(theta, x_expanded, t)

        return NormalizingFlow(
            transform=FreeFormJacobianTransform(
                f=vector_field,
                t0=x.new_tensor(0.0),
                t1=x.new_tensor(1.0),
                phi=(x, *self.parameters()), # 追踪梯度
            ),
            base=DiagNormal(self.zeros, self.ones).expand(x.shape[:-1]),
        )

def train_mmdit(trainset, 
          validset,
          prior,
          epochs,
          num_obs,
          # --- MM-DiT 专属超参数 ---
          hidden_dim=64,
          depth=4,
          heads=4,
          context_num_tokens=4,
          theta_num_tokens=4,
          # -----------------------
          initial_lr=1e-3,
          weight_decay=1e-2, 
          clip=1.0,
          use_wandb=False,
          save_path=None):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))

    if use_wandb:
        wandb.login()
        wandb.init(project="betapic_mmdit") # 更新项目名称
        config = {
            "epochs": epochs,
            "num_obs": num_obs,
            "architecture": "MM-DiT FMPE", # 记录架构
            "hidden_dim": hidden_dim,
            "depth": depth,
            "heads": heads,
            "initial_lr": initial_lr,
            "weight_decay": weight_decay,
            "clip": clip,
        }
        wandb.config.update(config)

    # 1. 初始化 MM-DiT FMPE Estimator
    # 注意：这里不再使用 NPE，而是使用你的自定义类
    estimator = MMDiTFMPE(
        theta_dim=8,           # BetaPic 的 8 个轨道参数
        context_dim=num_obs,   # 观测数据维度
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        context_num_tokens=context_num_tokens,
        theta_num_tokens=theta_num_tokens,
        is_individual=False    # 通常设为 False，除非你做单独参数推断
    ).to(device)
    
    # 2. 使用 Flow Matching Loss
    loss = FMPELoss(estimator)
    total_params = sum(p.numel() for p in estimator.parameters() if p.requires_grad)
    print(f"\n[Model Info] Total Trainable Parameters: {total_params:,}")
    print(f"[Model Info] Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (Assuming float32)\n")
    optimizer = optim.AdamW(
        estimator.parameters(), 
        lr=initial_lr, 
        weight_decay=weight_decay
    )
    
    step = GDStep(optimizer, clip=clip) 

    scheduler = sched.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        min_lr=1e-6,
        patience=32,
        threshold=1e-2,
        threshold_mode='abs'
    )

    with tqdm(range(epochs), unit='epoch') as tq:
        for epoch in tq:
            estimator.train()
            
            # FMPELoss 和 NPELoss 的调用接口是一样的: loss(theta, x)
            # 注意：prior.pre_process(theta) 非常重要，Flow Matching 对数据标准化很敏感
            train_loss = torch.stack([
                step(loss(prior.pre_process(theta).to(device), x.to(device))) 
                for theta, x in islice(trainset, 1024) 
            ]).cpu().numpy()
            
            estimator.eval()
            
            with torch.no_grad():
                valid_loss = torch.stack([
                    loss(prior.pre_process(theta).to(device), x.to(device))
                    for theta, x in islice(validset, 256) 
                ]).cpu().numpy()
            
            if use_wandb:
                wandb.log({
                    "train_loss": train_loss.mean(), 
                    "valid_loss": valid_loss.mean(), 
                    "lr": optimizer.param_groups[0]['lr']
                })
            
            scheduler.step(valid_loss.mean())

            if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
                break

            tq.set_postfix(loss=train_loss.mean(), val_loss=valid_loss.mean())

    # 保存模型
    if save_path is None:
        save_path = f"models/{wandb.run.name}.pth" if use_wandb else "models/betapic_mmdit.pth"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(estimator.state_dict(), str(save_path)) # 建议只保存 state_dict，更加稳健
    print(f"Model saved to {save_path}")
    
    if use_wandb:
        wandb.finish()
