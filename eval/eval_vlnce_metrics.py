import os
import json
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm

def compute_metrics(predictions, ground_truths):
    """
    计算 VLN-CE 的四个核心指标：SR, NE, SPL, TL
    :param predictions: 预测的轨迹列表
    :param ground_truths: 真实的轨迹列表
    """
    assert len(predictions) == len(ground_truths), "预测数量与真实数据不一致"
    
    total_episodes = len(predictions)
    success_count = 0
    total_ne = 0.0
    total_spl = 0.0
    total_tl = 0.0
    
    # 成功判定的距离阈值 (通常为 3 米)
    SUCCESS_RADIUS = 3.0
    
    for i in range(total_episodes):
        pred_traj = predictions[i]['trajectory']
        gt_traj = ground_truths[i]['trajectory']
        
        # 1. Trajectory Length (TL)
        # 计算预测轨迹的总长度
        tl = 0.0
        for j in range(1, len(pred_traj)):
            # 假设每个点是一个 (x, y, z) 坐标
            p1 = np.array(pred_traj[j-1])
            p2 = np.array(pred_traj[j])
            tl += np.linalg.norm(p2 - p1)
        total_tl += tl
        
        # 2. Navigation Error (NE)
        # 预测的终点与真实目标的欧氏距离
        pred_end = np.array(pred_traj[-1])
        gt_end = np.array(gt_traj[-1])
        ne = np.linalg.norm(pred_end - gt_end)
        total_ne += ne
        
        # 3. Success Rate (SR)
        # 终点距离目标小于 3 米
        is_success = int(ne <= SUCCESS_RADIUS)
        success_count += is_success
        
        # 4. Success weighted by Path Length (SPL)
        # SPL = S * (l / max(p, l))
        # l: 最短路径长度 (Ground Truth)
        # p: 实际走过的路径长度 (TL)
        l = 0.0
        for j in range(1, len(gt_traj)):
            g1 = np.array(gt_traj[j-1])
            g2 = np.array(gt_traj[j])
            l += np.linalg.norm(g2 - g1)
            
        spl = is_success * (l / max(tl, l)) if max(tl, l) > 0 else 0.0
        total_spl += spl
        
    metrics = {
        "SR": (success_count / total_episodes) * 100.0,
        "NE": total_ne / total_episodes,
        "SPL": (total_spl / total_episodes) * 100.0,
        "TL": total_tl / total_episodes
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate PAPlanner on VLN-CE metrics")
    parser.add_argument("--pred_file", type=str, required=True, help="JSON file containing predicted trajectories")
    parser.add_argument("--gt_file", type=str, required=True, help="JSON file containing ground truth trajectories")
    args = parser.parse_args()
    
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
        
    with open(args.gt_file, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
        
    # 确保根据 episode_id 对齐
    predictions = sorted(predictions, key=lambda x: x['episode_id'])
    ground_truths = sorted(ground_truths, key=lambda x: x['episode_id'])
    
    print(f"Evaluating {len(predictions)} episodes...")
    metrics = compute_metrics(predictions, ground_truths)
    
    print("\n--- VLN-CE Evaluation Results ---")
    print(f"Trajectory Length (TL) : {metrics['TL']:.2f}")
    print(f"Navigation Error (NE)  : {metrics['NE']:.2f}")
    print(f"Success Rate (SR)      : {metrics['SR']:.1f}%")
    print(f"SPL                    : {metrics['SPL']:.1f}%")
    print("---------------------------------\n")

if __name__ == "__main__":
    main()
