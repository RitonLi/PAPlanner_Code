import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .paplanner_head import PAPlannerHead

class PAPlannerModel(nn.Module):
    def __init__(self, vlm_model, hidden_size, num_action_types=4, num_distances=10):
        """
        包装 Qwen3-VL 模型，添加 PAPlanner 的定制化预测头和损失函数
        """
        super().__init__()
        self.vlm = vlm_model # Qwen3-VL backbone
        self.paplanner_head = PAPlannerHead(hidden_size, num_action_types, num_distances)
        
        
        self.lambda_1 = 1.0
        self.lambda_2 = 1.0
        self.lambda_3 = 1.0

    def forward(self, input_ids, images, sub_inst_embs, 
                gt_prog=None, gt_comp=None, gt_type=None, gt_dist=None,
                attention_mask=None, **kwargs):
        """
        前向传播，计算预测结果和损失
        """
        # 1. 提取 VLM 的隐藏层状态
        # 注意：这里我们不需要 VLM 生成文本，只需要它的 hidden_states
        outputs = self.vlm(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        # 获取最后一层的 hidden_states
        hidden_states = outputs.hidden_states[-1]
        
        # 获取序列中最后一个 token 的表示作为 h_t
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            h_t = hidden_states[batch_indices, sequence_lengths]
        else:
            h_t = hidden_states[:, -1, :] # [Batch, Hidden_Dim]
            
        # 2. 通过 PAPlanner 预测头
        prog_logits, comp_score, type_logits, dist_logits, pred_z_t = self.paplanner_head(
            h_t, sub_inst_embs, gt_z_t=gt_prog
        )
        
        # 3. 计算损失 (如果提供了 Ground Truth)
        loss = None
        loss_dict = {}
        if gt_prog is not None:
            # L_prog: 子指令进度预测损失 (Cross Entropy)
            loss_prog = nn.CrossEntropyLoss()(prog_logits, gt_prog)
            
            # L_comp: 完成度预测损失 (MSE)
            loss_comp = nn.MSELoss()(comp_score, gt_comp)
            
            # L_type: 动作类型预测损失 (Cross Entropy)
            loss_type = nn.CrossEntropyLoss()(type_logits, gt_type)
            
            # L_dist: 前进距离预测损失 (仅当动作类型为 Forward 时计算)
            # 假设 Forward 对应的 index 为 0
            forward_mask = (gt_type == 0)
            if forward_mask.sum() > 0:
                loss_dist = nn.CrossEntropyLoss()(dist_logits[forward_mask], gt_dist[forward_mask])
            else:
                loss_dist = torch.tensor(0.0, device=dist_logits.device)
                
            # 总损失
            loss = loss_prog + self.lambda_1 * loss_comp + self.lambda_2 * loss_type + self.lambda_3 * loss_dist
            
            loss_dict = {
                "loss": loss,
                "loss_prog": loss_prog,
                "loss_comp": loss_comp,
                "loss_type": loss_type,
                "loss_dist": loss_dist
            }
            
        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "prog_logits": prog_logits,
            "comp_score": comp_score,
            "type_logits": type_logits,
            "dist_logits": dist_logits,
            "pred_z_t": pred_z_t
        }
