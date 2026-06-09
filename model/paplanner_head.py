import torch
import torch.nn as nn

class PAPlannerHead(nn.Module):
    def __init__(self, hidden_size, num_action_types=4, num_distances=10):
        """
        PAPlanner 核心预测头
        :param hidden_size: VLM 输出的隐藏层维度
        :param num_action_types: 动作类型数量 (默认4: Forward, Turn left, Turn right, Stop)
        :param num_distances: 离散化的前进距离类别数量 (对应集合 D)
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. 子指令进度追踪 (Sub-instruction Progress Tracking)
        # 用于计算 h_t 和 S_emb 之间的相似度，预测当前执行的子指令索引 z_t
        self.prog_proj_h = nn.Linear(hidden_size, hidden_size)
        self.prog_proj_s = nn.Linear(hidden_size, hidden_size)
        
        # 完成度得分预测 (Completion Score g_t)
        self.comp_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 2. 动作类型预测 (Structured Action Prediction)
        self.type_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_action_types)
        )
        
        # 3. 前进距离预测 (Forward Distance Prediction)
        self.dist_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_distances)
        )

    def forward(self, h_t, S_emb, gt_z_t=None):
        """
        :param h_t: 当前时刻的视觉-语言联合表示 [Batch, Hidden_Dim]
        :param S_emb: 子指令集合的 Embedding [Batch, Num_Sub_Inst, Hidden_Dim]
        :param gt_z_t: 训练时使用的 Ground Truth 子指令索引 (可选)
        """
        # --- 进度追踪 ---
        h_proj = self.prog_proj_h(h_t).unsqueeze(1) # [B, 1, D]
        s_proj = self.prog_proj_s(S_emb)            # [B, M, D]
        
        # 计算每个子指令的匹配得分
        prog_logits = torch.sum(h_proj * s_proj, dim=-1) # [B, M]
        
        # 确定当前相关的子指令索引
        if self.training and gt_z_t is not None:
            z_t = gt_z_t # 训练时使用 Teacher Forcing
        else:
            z_t = torch.argmax(prog_logits, dim=-1) # 推理时使用预测值
            
        # 获取选中子指令的 Embedding (e_{z_t})
        batch_indices = torch.arange(h_t.size(0), device=h_t.device)
        e_z = S_emb[batch_indices, z_t] # [B, D]
        
        # 拼接 h_t 和 e_{z_t}
        combined_features = torch.cat([h_t, e_z], dim=-1) # [B, 2D]
        
        # --- 预测 ---
        comp_score = self.comp_mlp(combined_features).squeeze(-1) # [B]
        type_logits = self.type_mlp(combined_features)            # [B, 4]
        dist_logits = self.dist_mlp(combined_features)            # [B, K]
        
        return prog_logits, comp_score, type_logits, dist_logits, z_t
