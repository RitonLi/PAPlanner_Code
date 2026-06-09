import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEmbedder(nn.Module):
    """
    用于提取子指令 Embedding 的文本编码器
    论文中提到: e_{z_t} is the embedding of the predicted sub-instruction
    为了保持和 VLM 特征维度的对齐，这里我们使用一个预训练的文本模型，并加上一个投影层映射到 hidden_size
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", hidden_size=4096):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        
        # 冻结文本模型参数
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        # 投影层，将文本特征维度映射到 Qwen3-VL 的 hidden_size
        text_hidden_size = self.text_model.config.hidden_size
        self.proj = nn.Linear(text_hidden_size, hidden_size)
        
    def forward(self, texts, device="cuda"):
        """
        提取文本列表的 Embedding
        :param texts: 字符串列表 (如子指令列表)
        :return: [len(texts), hidden_size]
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            
        # 使用 [CLS] token 或者 Mean Pooling 作为句子表示
        # 这里使用 Mean Pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask # [N, text_hidden_size]
        
        # 映射到 VLM 的 hidden_size
        projected_embeddings = self.proj(sentence_embeddings) # [N, hidden_size]
        
        return projected_embeddings
