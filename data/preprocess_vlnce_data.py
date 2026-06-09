import json
import re
import argparse
from tqdm import tqdm

def split_instruction(instruction):
    """
    根据论文描述，使用显式的语言线索（如 'and', 'then', 'after', ','）将长指令切分为子指令。
    """
    # 定义切分关键词
    delimiters = r',| and | then | after | \. '
    
    # 切分并去除空白字符
    sub_insts = [s.strip() for s in re.split(delimiters, instruction) if s.strip()]
    
    # 过滤掉过短的无意义片段
    sub_insts = [s for s in sub_insts if len(s.split()) > 1]
    
    if not sub_insts:
        sub_insts = [instruction] # 如果切分失败，保留原指令
        
    return sub_insts

def calculate_completion_score(current_step, total_steps):
    """
    计算完成度得分 g_t (0到1之间)
    论文中提到：基于已遍历路径长度与当前局部轨迹总长度的比例
    这里用步数比例简化模拟
    """
    if total_steps <= 1:
        return 1.0
    return min(1.0, current_step / (total_steps - 1))

def process_vlnce_trajectory(trajectory, history_len=8):
    """
    处理单条 VLN-CE 轨迹，生成训练样本
    """
    samples = []
    
    instruction = trajectory['instruction']
    sub_instructions = split_instruction(instruction)
    
    path = trajectory['path'] # 包含每一步的图像、动作、距离等
    total_steps = len(path)
    
    # 模拟子指令的对齐 (这里假设均匀分配，实际中你可能需要用 LLM 标注或基于地标对齐)
    steps_per_sub = max(1, total_steps // len(sub_instructions))
    
    history_frames = []
    
    for t, step in enumerate(path):
        # 维护历史帧
        history_frames.append(step['image_path'])
        if len(history_frames) > history_len:
            history_frames.pop(0)
            
        # 确定当前相关的子指令索引 z_t
        z_t = min(t // steps_per_sub, len(sub_instructions) - 1)
        
        # 计算完成度 g_t
        # 假设当前子指令的起始步和结束步
        start_step = z_t * steps_per_sub
        end_step = min((z_t + 1) * steps_per_sub, total_steps)
        local_total = end_step - start_step
        local_current = t - start_step
        g_t = calculate_completion_score(local_current, local_total)
        
        # 获取动作类型和前进距离
        action_type = step['action_type'] # "Forward", "Turn left", "Turn right", "Stop"
        forward_distance = step.get('forward_distance', 0.0) # 如果是 Forward，获取距离
        
        # 聚合时间上相邻的相似 Forward 动作 (论文中的优化点)
        # 这里简化处理，直接使用当前步的动作
        
        sample = {
            "instruction": instruction,
            "sub_instructions": sub_instructions,
            "image_paths": list(history_frames), # 历史帧 + 当前帧
            "current_sub_instruction_idx": z_t,
            "completion_score": g_t,
            "action_type": action_type,
            "forward_distance": forward_distance
        }
        samples.append(sample)
        
    return samples

def main():
    parser = argparse.ArgumentParser(description="Preprocess VLN-CE data for PAPlanner")
    parser.add_argument("--input_json", type=str, required=True, help="Raw VLN-CE trajectory data")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Processed training data for PAPlanner")
    args = parser.parse_args()
    
    with open(args.input_json, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    processed_samples = []
    for traj in tqdm(raw_data, desc="Processing trajectories"):
        samples = process_vlnce_trajectory(traj)
        processed_samples.extend(samples)
        
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for sample in processed_samples:
            f.write(json.dumps(sample) + '\n')
            
    print(f"Successfully processed {len(raw_data)} trajectories into {len(processed_samples)} training samples.")
    print(f"Saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()
