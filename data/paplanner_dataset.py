import torch
import json
from torch.utils.data import Dataset
from PIL import Image
from qwen3_vl.model.text_embedder import TextEmbedder

class PAPlannerDataset(Dataset):
    def __init__(self, data_path, tokenizer, image_processor, max_sub_instructions=5, max_length=1024, hidden_size=4096):
        """
        PAPlanner 数据集加载器
        :param data_path: 包含导航轨迹和子指令标签的 JSONL/JSON 文件路径
        :param tokenizer: Qwen3-VL Tokenizer
        :param image_processor: 图像预处理器
        :param max_sub_instructions: 最大的子指令数量
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_sub_instructions = max_sub_instructions
        self.max_length = max_length
        
        # 动作类型映射 (与模型预测头对应)
        self.action_type_map = {
            "Forward": 0,
            "Turn left": 1,
            "Turn right": 2,
            "Stop": 3
        }
        
        # 前进距离离散化集合 (对应论文中的 D 集合)
        self.distance_set = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
        
        # 文本编码器 (用于提取子指令的 Embedding)
        self.text_encoder = TextEmbedder(hidden_size=hidden_size)
        self.text_encoder.eval()

    def _load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def _get_closest_distance_idx(self, dist):
        # 找到最接近的离散距离索引
        distances = torch.tensor(self.distance_set)
        return torch.argmin(torch.abs(distances - dist)).item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 图像处理 (历史帧 + 当前帧)
        image_paths = item['image_paths'] # 包含历史观测和当前观测
        images = [Image.open(p).convert('RGB') for p in image_paths]
        # 使用 image_processor 处理图像
        # images_tensor = self.image_processor(images, return_tensors='pt')
        
        # 2. 文本指令处理
        full_instruction = item['instruction']
        sub_instructions = item['sub_instructions'] # 列表形式的子指令
        
        # 截断或填充子指令列表
        if len(sub_instructions) > self.max_sub_instructions:
            sub_instructions = sub_instructions[:self.max_sub_instructions]
        else:
            sub_instructions += [""] * (self.max_sub_instructions - len(sub_instructions))
            
        # 提取子指令 Embedding
        with torch.no_grad():
            sub_inst_embs = self.text_encoder(sub_instructions, device="cpu") # [max_sub_instructions, hidden_size]
        
        # 3. 构造 Prompt
        # 对应论文中的 Navigation Prompts
        prompt = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f"of historical observations, and current observation. Your assigned task is: '{full_instruction}' "
            f"Analyze this series of images to decide your next action."
        )
        
        # Tokenize prompt
        # input_ids = self.tokenizer(prompt, max_length=self.max_length, truncation=True, return_tensors="pt")
        
        # 4. 获取 Ground Truth 标签
        # 进度追踪标签 (当前执行的子指令索引)
        gt_prog = item['current_sub_instruction_idx']
        
        # 完成度标签 (0到1之间)
        gt_comp = item['completion_score']
        
        # 动作类型标签
        action_str = item['action_type']
        gt_type = self.action_type_map.get(action_str, 3) # 默认 Stop
        
        # 前进距离标签 (仅当 action_type 为 Forward 时有效)
        gt_dist = 0
        if gt_type == 0:
            gt_dist = self._get_closest_distance_idx(item.get('forward_distance', 0.0))
            
        return {
            # "input_ids": input_ids.squeeze(0),
            # "images": images_tensor,
            "sub_inst_embs": sub_inst_embs,
            "gt_prog": torch.tensor(gt_prog, dtype=torch.long),
            "gt_comp": torch.tensor(gt_comp, dtype=torch.float32),
            "gt_type": torch.tensor(gt_type, dtype=torch.long),
            "gt_dist": torch.tensor(gt_dist, dtype=torch.long)
        }
