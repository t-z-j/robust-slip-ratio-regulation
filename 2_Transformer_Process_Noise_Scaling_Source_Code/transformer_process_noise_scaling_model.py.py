import torch
import torch.nn as nn


WINDOW_SIZE = 100
FEATURE_DIM = 30
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
REGRESSION_DIM = 32
DROPOUT = 0.08
OUTPUT_DIM = 1
Q_SCALE_MIN = 1.0
Q_SCALE_MAX = 20.0

FEATURE_NAMES = (
    "time_s",
    "v_meas_kmh",
    "v_ekf_kmh",
    "x_pred_kmh",
    "innovation_kmh",
    "innovation_used_kmh",
    "K",
    "P",
    "P_pred",
    "F_k",
    "Q_adapt",
    "Q_used",
    "Q_hat",
    "R",
    "Fx_total",
    "a_dyn",
    "rpm_L1_fl",
    "rpm_R1_fr",
    "rpm_L2_rl",
    "rpm_R2_rr",
    "slip_front_pct",
    "slip_rear_pct",
    "T_L1_fl",
    "T_R1_fr",
    "T_L2_rl",
    "T_R2_rr",
    "Fz_L1_fl",
    "Fz_R1_fr",
    "Fz_L2_rl",
    "Fz_R2_rr",
)

# Paper-style symbols corresponding one-to-one to the actual training inputs above.
# The wheel-speed channels are stored in rpm, and the two slip features are
# front- and rear-axle averages expressed in percent.
PAPER_FEATURE_SYMBOLS = (
    "t",
    "v_CAN",
    "v_post",
    "v_prior",
    "innovation",
    "innovation_used",
    "K",
    "P_post",
    "P_pred",
    "F",
    "Q_adapt",
    "Q_used",
    "Q_hat",
    "R",
    "F_x_net",
    "a_x",
    "omega_fl_rpm",
    "omega_fr_rpm",
    "omega_rl_rpm",
    "omega_rr_rpm",
    "lambda_front_avg_pct",
    "lambda_rear_avg_pct",
    "T_fl",
    "T_fr",
    "T_rl",
    "T_rr",
    "Fz_fl",
    "Fz_fr",
    "Fz_rl",
    "Fz_rr",
)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attention)
        return self.norm2(x + self.ff(x))


class TractorAKITQModel(nn.Module):
    def __init__(self, feature_mean, feature_std):
        super(TractorAKITQModel, self).__init__()
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(feature_mean, dtype=torch.float32).reshape(1, 1, FEATURE_DIM),
        )
        self.register_buffer(
            "feature_std",
            torch.as_tensor(feature_std, dtype=torch.float32).reshape(1, 1, FEATURE_DIM),
        )

        self.input_proj = nn.Linear(FEATURE_DIM, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW_SIZE, D_MODEL))
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    D_MODEL,
                    NUM_HEADS,
                    DIM_FEEDFORWARD,
                    DROPOUT,
                )
                for _ in range(NUM_LAYERS)
            ]
        )
        self.seed = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pool = nn.MultiheadAttention(
            D_MODEL,
            NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, REGRESSION_DIM),
            nn.GELU(),
            nn.Linear(REGRESSION_DIM, OUTPUT_DIM),
        )

    def forward(self, feature_window):
        # feature_window shape: (batch_size, 100, 30)
        x = (feature_window - self.feature_mean) / self.feature_std
        x = self.input_proj(x) + self.pos

        for block in self.blocks:
            x = block(x)

        query = self.seed.repeat(x.shape[0], 1, 1)
        pooled, _ = self.pool(query, x, x, need_weights=False)
        log_q_scale = self.head(pooled[:, 0, :])
        return log_q_scale


def log_q_scale_to_q_scale(log_q_scale):
    return torch.clamp(
        torch.exp(log_q_scale),
        min=Q_SCALE_MIN,
        max=Q_SCALE_MAX,
    )