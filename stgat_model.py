"""
stgat_model.py — Spatio-Temporal Graph Attention Network 預測模型

架構（對應論文 Section 4.2）：
  ┌─────────────────────────────────────────────────┐
  │  Node Stream (需求 / 空車)                       │
  │    Path-Fixed   : [GTCN → GAT(A_fixed, edge)]×L │
  │    Path-Adaptive: [GTCN → GAT(A_adp)]×L         │
  │    → GatedFusion → demand_head / supply_head     │
  ├─────────────────────────────────────────────────┤
  │  Edge Stream (道路速度)                          │
  │    [GTCN → GAT(line_graph)]×L                    │
  │    → speed_head                                  │
  └─────────────────────────────────────────────────┘

輸出：predicted demand, vacant taxis, road speeds for next p steps
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
#  工具函式
# ════════════════════════════════════════════════════════════════

def build_line_graph_adj(edge_index: torch.Tensor) -> torch.Tensor:
    """
    從邊列表建構線圖 (line graph) 鄰接矩陣。
    兩條有向邊 (i,j) 與 (u,v) 相鄰 iff j == u（物理連續）。
    """
    dst = edge_index[:, 1]  # (|E|,)
    src = edge_index[:, 0]  # (|E|,)
    return (dst.unsqueeze(1) == src.unsqueeze(0)).float()


# ════════════════════════════════════════════════════════════════
#  GTCN — Gated Temporal Convolutional Network
# ════════════════════════════════════════════════════════════════

class GTCNLayer(nn.Module):
    """
    單層門控時間卷積（Eq. 1-2）。
    output = gate ⊗ filter + (1 - gate) ⊗ residual
    其中 gate = σ(conv(x))
    """

    def __init__(
        self, in_c: int, out_c: int, kernel_size: int = 3, dilation: int = 1
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.gate_conv = nn.Conv1d(in_c, out_c, kernel_size, dilation=dilation)
        self.filter_conv = nn.Conv1d(in_c, out_c, kernel_size, dilation=dilation)
        self.skip_conv = nn.Conv1d(in_c, out_c, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, C_in) → (B, N, T, C_out)"""
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).permute(0, 2, 1)       # (B*N, C, T)
        x = F.pad(x, (self.pad, 0))                         # causal left-pad
        gate = torch.sigmoid(self.gate_conv(x))
        out = gate * self.filter_conv(x) + (1 - gate) * self.skip_conv(x)
        return out.permute(0, 2, 1).reshape(B, N, T, -1)


class GTCN(nn.Module):
    """多層 GTCN，含殘差連接與 LayerNorm。"""

    def __init__(
        self,
        in_c: int,
        hid_c: int,
        out_c: int,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            ic = in_c if i == 0 else hid_c
            oc = hid_c if i < num_layers - 1 else out_c
            self.layers.append(GTCNLayer(ic, oc, kernel_size, dilation=2 ** i))
            self.norms.append(nn.LayerNorm(oc))
        self.skip = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, C_in) → (B, N, T, C_out)"""
        res = self.skip(x)
        for layer, norm in zip(self.layers, self.norms):
            x = F.relu(norm(layer(x)))
        return x + res


# ════════════════════════════════════════════════════════════════
#  GAT — Graph Attention Layer
# ════════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    """
    通用圖注意力層，支援可選的邊特徵（Eq. 3-7 / 8-12）。

    若 edge_in > 0 → 區域級 GAT（含邊特徵 len, speed）
    若 edge_in == 0 → 邊級 GAT（不含額外邊特徵）
    """

    def __init__(
        self,
        in_c: int,
        d_out: int,
        num_heads: int = 4,
        edge_in: int = 0,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.concat = concat
        self.has_edge = edge_in > 0
        total = d_out * num_heads

        self.W_q = nn.Linear(in_c, total, bias=False)
        self.W_k = nn.Linear(in_c, total, bias=False)
        self.W_v = nn.Linear(in_c, total, bias=False)

        self.a_q = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        self.a_k = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        nn.init.xavier_normal_(self.a_q)
        nn.init.xavier_normal_(self.a_k)

        if self.has_edge:
            self.W_e = nn.Linear(edge_in, total, bias=False)
            self.a_e = nn.Parameter(torch.empty(1, 1, 1, num_heads, d_out))
            nn.init.xavier_normal_(self.a_e)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h         : (B, N, C_in)
        adj       : (N, N) binary mask
        edge_feat : (B, N, N, edge_in) or None
        return    : (B, N, H*d_out) if concat else (B, N, d_out)
        """
        B, N, _ = h.shape
        H, D = self.num_heads, self.d_out

        q = self.W_q(h).view(B, N, H, D)
        k = self.W_k(h).view(B, N, H, D)
        v = self.W_v(h).view(B, N, H, D)

        # additive attention: a_q·q_i + a_k·k_j (+ a_e·e_{i,j})
        s_q = (q * self.a_q).sum(-1)  # (B, N, H)
        s_k = (k * self.a_k).sum(-1)  # (B, N, H)
        scores = s_q.unsqueeze(2) + s_k.unsqueeze(1)  # (B, N, N, H)

        if self.has_edge and edge_feat is not None:
            e = self.W_e(edge_feat).view(B, N, N, H, D)
            s_e = (e * self.a_e).sum(-1)  # (B, N, N, H)
            scores = scores + s_e

        scores = F.leaky_relu(scores, 0.2)

        # mask non-neighbors
        mask = adj.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        scores = scores.masked_fill(mask == 0, -1e9)

        alpha = F.softmax(scores, dim=2)  # (B, N, N, H)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # aggregate: s'_i = Σ_j α_{i,j} · v_j
        out = torch.einsum("bnjh,bnjhd->bnhd", alpha, v.unsqueeze(1).expand(-1, N, -1, -1, -1))

        if self.concat:
            out = out.reshape(B, N, H * D)
        else:
            out = out.mean(dim=2)

        return F.elu(out)


# ════════════════════════════════════════════════════════════════
#  Gated Fusion（Eq. 13-14）
# ════════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)
        self.W3 = nn.Linear(dim, dim)

    def forward(self, h_fixed: torch.Tensor, h_adp: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.W1(h_fixed + h_adp))
        return torch.tanh(gate * self.W2(h_fixed) + (1 - gate) * self.W3(h_adp))


# ════════════════════════════════════════════════════════════════
#  STGATPredictor — 完整預測模型
# ════════════════════════════════════════════════════════════════

class STGATPredictor(nn.Module):
    """
    Parameters
    ----------
    num_nodes       : N — 區域數
    edge_index      : (|E|, 2) long — 有向邊列表
    edge_lengths    : (|E|,) float — 靜態路長
    adj_matrix      : (N, N) float — 物理鄰接 (0/1)
    hidden_dim      : 隱藏層維度（也是 GTCN 輸出與 GAT 的 heads × d_out）
    num_heads       : 注意力頭數
    num_st_blocks   : ST-Block 堆疊數
    pred_horizon    : 預測未來 p 步
    hist_len        : 歷史窗口長度 h（僅用於說明，不影響模型定義）
    adaptive_emb    : 自適應鄰接 embedding 維度
    """

    def __init__(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        adj_matrix: torch.Tensor,
        *,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_st_blocks: int = 2,
        num_gtcn_layers: int = 2,
        kernel_size: int = 3,
        pred_horizon: int = 3,
        adaptive_emb: int = 10,
        node_feat_dim: int = 2,
        edge_feat_dim: int = 2,
    ) -> None:
        super().__init__()
        N = num_nodes
        d_per_head = hidden_dim // num_heads
        assert hidden_dim == d_per_head * num_heads, "hidden_dim must be divisible by num_heads"

        # ── 圖結構（不需要梯度） ──
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_lengths", edge_lengths.float())
        adj_with_self = (adj_matrix + torch.eye(N)).clamp(max=1.0)
        self.register_buffer("adj_fixed", adj_with_self)
        self.register_buffer("adj_full", torch.ones(N, N))
        self.register_buffer("line_adj", build_line_graph_adj(edge_index))

        # ── 自適應鄰接 ──
        self.emb_src = nn.Parameter(torch.randn(N, adaptive_emb) * 0.1)
        self.emb_dst = nn.Parameter(torch.randn(N, adaptive_emb) * 0.1)

        # ── 輸入投影 ──
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(1, hidden_dim)

        # ── Node path — fixed topology ──
        self.n_gtcn_fix = nn.ModuleList()
        self.n_gat_fix = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.n_gtcn_fix.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.n_gat_fix.append(
                GATLayer(hidden_dim, d_per_head, num_heads, edge_in=edge_feat_dim, concat=True)
            )

        # ── Node path — adaptive topology ──
        self.n_gtcn_adp = nn.ModuleList()
        self.n_gat_adp = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.n_gtcn_adp.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.n_gat_adp.append(
                GATLayer(hidden_dim, d_per_head, num_heads, edge_in=0, concat=True)
            )

        # ── Fusion ──
        self.fusion = GatedFusion(hidden_dim)

        # ── Edge path ──
        self.e_gtcn = nn.ModuleList()
        self.e_gat = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.e_gtcn.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.e_gat.append(
                GATLayer(hidden_dim, d_per_head, num_heads, edge_in=0, concat=True)
            )

        # ── 輸出頭 ──
        self.demand_head = nn.Linear(hidden_dim, pred_horizon)
        self.supply_head = nn.Linear(hidden_dim, pred_horizon)
        self.speed_head = nn.Linear(hidden_dim, pred_horizon)

    # ── helpers ────────────────────────────────────────────────

    def _adaptive_adj(self) -> torch.Tensor:
        return F.softmax(F.relu(self.emb_src @ self.emb_dst.T), dim=1)

    def _edge_feat_at(self, speed: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        speed : (BT, |E|)  — 每條邊在某時間步的速度
        return: (BT, N, N, 2) — [road_length, speed]
        """
        BT = speed.shape[0]
        N = self.adj_fixed.shape[0]
        src, dst = self.edge_index[:, 0], self.edge_index[:, 1]
        feat = torch.zeros(BT, N, N, 2, device=device)
        feat[:, src, dst, 0] = self.edge_lengths.unsqueeze(0).expand(BT, -1)
        feat[:, src, dst, 1] = speed
        return feat

    def _run_node_path(
        self,
        node_h: torch.Tensor,
        speed_seq: torch.Tensor,
        gtcn_list: nn.ModuleList,
        gat_list: nn.ModuleList,
        adj: torch.Tensor,
        use_edge_feat: bool,
    ) -> torch.Tensor:
        """回傳最後時間步的節點表示 (B, N, hidden_dim)"""
        B, N, T, _ = node_h.shape
        x = node_h
        for gtcn, gat in zip(gtcn_list, gat_list):
            x = gtcn(x)                                              # (B, N, T, C)
            x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)    # (BT, N, C)
            if use_edge_feat:
                sp_flat = speed_seq.permute(0, 2, 1).reshape(B * T, -1)
                ef = self._edge_feat_at(sp_flat, x.device)
            else:
                ef = None
            x_flat = gat(x_flat, adj, ef)                            # (BT, N, C)
            x = x_flat.reshape(B, T, N, -1).permute(0, 2, 1, 3)     # (B, N, T, C)
        return x[:, :, -1, :]                                        # (B, N, C)

    # ── forward ───────────────────────────────────────────────

    def forward(
        self, node_seq: torch.Tensor, speed_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        node_seq  : (B, N, T, 2) — [demand, vacant] 歷史序列
        speed_seq : (B, |E|, T)  — 歷史邊速度

        Returns
        -------
        demand_pred : (B, N, p)
        supply_pred : (B, N, p)
        speed_pred  : (B, |E|, p)
        """
        B, N, T, _ = node_seq.shape
        nE = speed_seq.shape[1]

        node_h = self.node_proj(node_seq)                            # (B, N, T, C)
        edge_h = self.edge_proj(speed_seq.unsqueeze(-1))             # (B, |E|, T, C)

        # ── Node: fixed path ──
        h_fix = self._run_node_path(
            node_h, speed_seq,
            self.n_gtcn_fix, self.n_gat_fix, self.adj_fixed,
            use_edge_feat=True,
        )

        # ── Node: adaptive path ──
        h_adp = self._run_node_path(
            node_h, speed_seq,
            self.n_gtcn_adp, self.n_gat_adp, self.adj_full,
            use_edge_feat=False,
        )

        # ── Fusion ──
        h_node = self.fusion(h_fix, h_adp)                          # (B, N, C)

        # ── Edge path ──
        x_e = edge_h
        for gtcn, gat in zip(self.e_gtcn, self.e_gat):
            x_e = gtcn(x_e)                                         # (B, |E|, T, C)
            x_flat = x_e.permute(0, 2, 1, 3).reshape(B * T, nE, -1)
            x_flat = gat(x_flat, self.line_adj)
            x_e = x_flat.reshape(B, T, nE, -1).permute(0, 2, 1, 3)
        h_edge = x_e[:, :, -1, :]                                   # (B, |E|, C)

        # ── 輸出 ──
        demand_pred = self.demand_head(h_node)                       # (B, N, p)
        supply_pred = self.supply_head(h_node)                       # (B, N, p)
        speed_pred = self.speed_head(h_edge)                         # (B, |E|, p)

        return demand_pred, supply_pred, speed_pred
