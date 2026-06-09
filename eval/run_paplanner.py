import argparse
import torch
import cv2
from PIL import Image
from transformers import AutoProcessor

from qwen3_vl.model.builder import load_pretrained_model
from qwen3_vl.model.paplanner_model import PAPlannerModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained PAPlanner model")
    parser.add_argument("--video-file", type=str, required=True, help="Path to the input video (historical + current frames)")
    parser.add_argument("--instruction", type=str, required=True, help="The full navigation instruction")
    parser.add_argument("--sub-instructions", type=str, nargs='+', required=True, help="List of sub-instructions")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载基础 VLM 模型
    print("Loading VLM backbone...")
    tokenizer, vlm_model, image_processor, context_len = load_pretrained_model(
        args.model_path, "qwen3_vl", None
    )
    
    # 2. 包装为 PAPlannerModel
    print("Initializing PAPlanner...")
    hidden_size = vlm_model.config.hidden_size
    model = PAPlannerModel(vlm_model, hidden_size=hidden_size)
    model.load_state_dict(torch.load(f"{args.model_path}/paplanner_head.pth"), strict=False)
    model.eval()
    model.cuda()
    
    # 3. 处理视频输入 (历史帧 + 当前帧)
    print(f"Processing video: {args.video_file}")
    vidcap = cv2.VideoCapture(args.video_file)
    frames = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(image))
        
    # 假设我们取最后 N 帧作为输入
    num_frames = 8
    if len(frames) > num_frames:
        frames = frames[-num_frames:]
        
    images_tensor = image_processor(frames, return_tensors='pt').to(model.vlm.device, dtype=torch.float16)
    
    # 4. 处理文本输入
    prompt = (
        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
        f"of historical observations, and current observation. Your assigned task is: '{args.instruction}' "
        f"Analyze this series of images to decide your next action."
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    
    # 提取子指令的 Embedding
    from qwen3_vl.model.text_embedder import TextEmbedder
    text_encoder = TextEmbedder(hidden_size=hidden_size).cuda()
    text_encoder.eval()
    with torch.no_grad():
        sub_inst_embs = text_encoder(args.sub_instructions, device="cuda").unsqueeze(0).to(dtype=torch.float16) # [1, num_sub_insts, hidden_size]
    
    # 5. 推理
    print("Running inference...")
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            images=images_tensor,
            sub_inst_embs=sub_inst_embs
        )
        
    # 6. 解析输出
    action_types = ["Forward", "Turn left", "Turn right", "Stop"]
    distances = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    
    # 获取预测的子指令索引
    pred_z_t = outputs["pred_z_t"].item()
    active_sub_inst = args.sub_instructions[pred_z_t]
    
    # 获取完成度
    comp_score = outputs["comp_score"].item()
    
    # 获取动作类型
    type_idx = torch.argmax(outputs["type_logits"], dim=-1).item()
    action_type = action_types[type_idx]
    
    # 获取前进距离
    if action_type == "Forward":
        dist_idx = torch.argmax(outputs["dist_logits"], dim=-1).item()
        distance = distances[dist_idx]
        final_command = f"Forward({distance}m)"
    else:
        final_command = action_type
        
    print("\n--- PAPlanner Prediction ---")
    print(f"Active Sub-instruction: [{pred_z_t}] {active_sub_inst}")
    print(f"Completion Score: {comp_score:.2f}")
    print(f"Action Type: {action_type}")
    print(f"Final Command: {final_command}")
    print("----------------------------\n")

if __name__ == "__main__":
    main()
