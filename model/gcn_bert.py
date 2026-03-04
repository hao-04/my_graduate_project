import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 28 markers 对应索引：
#  0: C7,     1: T10,    2: CLAV,   3: STRN,
#  4: L_ASI,  5: L_SHO,  6: L_ELB,  7: L_WRA,
#  8: L_WRB,  9: L_FIN, 10: L_THI, 11: L_KNE,
# 12: L_TIB, 13: L_ANK, 14: L_HEE, 15: L_TOE,
# 16: R_ASI, 17: R_SHO, 18: R_ELB, 19: R_WRA,
# 20: R_WRB, 21: R_FIN, 22: R_THI, 23: R_KNE,
# 24: R_TIB, 25: R_ANK, 26: R_HEE, 27: R_TOE
# ---------------------------------------------------------------------------

SKELETON_EDGES = [
    # ---- 躯干 / 脊柱 ----
    (0, 1),    # C7   – T10
    (0, 2),    # C7   – CLAV
    (2, 3),    # CLAV – STRN
    (1, 4),    # T10  – L_ASI
    (1, 16),   # T10  – R_ASI
    (4, 16),   # L_ASI – R_ASI
    # ---- 左臂 ----
    (2, 5),    # CLAV  – L_SHO
    (5, 6),    # L_SHO – L_ELB
    (6, 7),    # L_ELB – L_WRA
    (7, 8),    # L_WRA – L_WRB
    (7, 9),    # L_WRA – L_FIN
    # ---- 左腿 ----
    (4, 10),   # L_ASI – L_THI
    (10, 11),  # L_THI – L_KNE
    (11, 12),  # L_KNE – L_TIB
    (12, 13),  # L_TIB – L_ANK
    (13, 14),  # L_ANK – L_HEE
    (13, 15),  # L_ANK – L_TOE
    # ---- 右臂 ----
    (2, 17),   # CLAV  – R_SHO
    (17, 18),  # R_SHO – R_ELB
    (18, 19),  # R_ELB – R_WRA
    (19, 20),  # R_WRA – R_WRB
    (19, 21),  # R_WRA – R_FIN
    # ---- 右腿 ----
    (16, 22),  # R_ASI – R_THI
    (22, 23),  # R_THI – R_KNE
    (23, 24),  # R_KNE – R_TIB
    (24, 25),  # R_TIB – R_ANK
    (25, 26),  # R_ANK – R_HEE
    (25, 27),  # R_ANK – R_TOE
]


def get_adjacency_matrix(num_nodes: int = 28) -> np.ndarray:
    """构建对称归一化邻接矩阵  Â = D^{-1/2} (A + I) D^{-1/2}"""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # 加自环
    A = A + np.eye(num_nodes, dtype=np.float32)
    # 对称归一化
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


# ========================== GCN 部分 ==========================


class GraphConvolution(nn.Module):
    """单层图卷积：  output = A · X · W + b"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C_in)
        A: (N, N)  归一化邻接矩阵
        return: (B, N, C_out)
        """
        support = torch.matmul(x, self.weight)   # (B, N, C_out)
        output = torch.matmul(A, support)         # (B, N, C_out)
        return output + self.bias


class GCN(nn.Module):
    """
    3 层 GCN，对每一帧的关节进行空间特征聚合。
    输入:  (B, T, N, C_in)  = (batch, 1001, 28, 3)
    输出:  (B, T, N, C_out) = (batch, 1001, 28, 3)
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 64,
        out_features: int = 3,
        num_nodes: int = 28,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, hidden_features)
        self.gc3 = GraphConvolution(hidden_features, out_features)

        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)

        self.dropout = nn.Dropout(dropout)

        # 将归一化邻接矩阵注册为不参与梯度的 buffer
        A_norm = get_adjacency_matrix(num_nodes)
        self.register_buffer("A", torch.from_numpy(A_norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape

        # 保存最开始的输入
        x0 = x

        # 合并 batch 和 time 维度一起做图卷积
        x = x.reshape(B * T, N, C)

        # Layer 1
        x = self.gc1(x, self.A)                    # (B*T, N, hidden)
        x = x.transpose(1, 2)                      # (B*T, hidden, N)  — BN 需要 channel-first
        x = self.bn1(x)
        x = x.transpose(1, 2)                      # (B*T, N, hidden)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.gc2(x, self.A)
        x = x.transpose(1, 2)
        x = self.bn2(x)
        x = x.transpose(1, 2)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3（不加激活，保留线性输出）
        x = self.gc3(x, self.A)                    # (B*T, N, out)

        x = x.reshape(B, T, N, -1)
        
        return x + x0


# ======================== BERT 编码器部分 ========================


class PositionalEncoding(nn.Module):
    """正弦-余弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 1100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SkeletonBERT(nn.Module):
    """
    BERT 风格的 Transformer 编码器。
    在序列头部插入可学习的 [CLS] token，
    经过 3 层 Encoder 后取出 [CLS] 表示用于分类。
    """

    def __init__(
        self,
        d_model: int = 84,
        nhead: int = 6,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 1100,
    ):
        super().__init__()
        self.d_model = d_model

        # 可学习的 [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)  e.g. (batch, 1001, 84)
        return: (B, d_model) — [CLS] 向量
        """
        B = x.size(0)

        # 在序列最前面拼接 [CLS] token  →  (B, 1+T, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1002, 84)

        # 加入位置编码
        x = self.pos_encoder(x)

        # Transformer 编码
        x = self.encoder(x)  # (B, 1002, 84)

        # 取出 [CLS] 位置的输出
        cls_output = self.layer_norm(x[:, 0, :])  # (B, d_model)
        return cls_output


# ======================== 完整模型 ========================


class GCN_BERT(nn.Module):
    """
    GCN + BERT 骨骼序列分类模型。

    流程:
        1. 3 层 GCN  — 空间聚合
           (B, 1001, 28, 3) → (B, 1001, 28, 3)
        2. 展平关节维度
           (B, 1001, 28, 3) → (B, 1001, 84)
        3. BERT 编码器 — 时序建模 + [CLS]
           (B, 1001, 84)   → (B, 84)
        4. 分类器
           (B, 84)         → (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 2,
        gcn_hidden: int = 64,
        d_model: int = 84,
        nhead: int = 6,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ---- 空间模块：GCN ----
        self.gcn = GCN(
            in_features=3,
            hidden_features=gcn_hidden,
            out_features=3,
            num_nodes=28,
            dropout=dropout,
        )

        # ---- 扩展得到的特征向量维度：Extend Dimension----
        self.exted_dim = nn.Linear(84, d_model)

        # ---- 时序模块：BERT Encoder ----
        self.bert = SkeletonBERT(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1001, 28, 3)
        return: (B, num_classes)
        """
        # 1. GCN 空间聚合
        x = self.gcn(x)  # (B, 1001, 28, 3)

        # 2. 拼接关节坐标 28×3 → 84
        B, T, N, C = x.shape
        x = x.reshape(B, T, N * C)  # (B, 1001, 84)

        # 3. 扩展特征向量为d_model 84 → d_model
        x = self.exted_dim(x)

        # 4. BERT 编码 → 取 [CLS]
        cls_output = self.bert(x)  # (B, d_model)

        # 5. 分类
        logits = self.classifier(cls_output)  # (B, num_classes)
        return logits
