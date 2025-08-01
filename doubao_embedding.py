#from src.model import MMEBModel
#from src.arguments import ModelArguments
#from src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens
import os
import pickle
import httpx
from volcenginesdkarkruntime import Ark
from PIL import Image
import torch
import numpy as np
from definition import positive_definition, daoliu_definition, qizha_definition
import base64
from tqdm import tqdm
#cuda_device = torch.device("cuda")
import json

def compute_similarity(qry, tar, cuda=False):
    """
    计算查询向量和目标向量之间的余弦相似度。

    参数:
    qry (torch.Tensor): 查询向量，形状可以是 (N, D) 或 (D,)。
    tar (torch.Tensor): 目标向量，形状可以是 (M, D) 或 (D,)。

    返回:
    torch.Tensor: 余弦相似度矩阵，形状为 (N, M) 或 (1,) 如果两个输入都是一维向量。
    """
    # 对查询向量和目标向量进行 L2 归一化处理
    if cuda == False: 
        qry = torch.tensor(qry, dtype=torch.float32)
        tar = torch.tensor(tar, dtype=torch.float32)
        qry_norm = torch.nn.functional.normalize(qry, p=2, dim=-1)
        tar_norm = torch.nn.functional.normalize(tar, p=2, dim=-1)

        # 计算归一化后向量的点积得到余弦相似度
        similarity = torch.matmul(qry_norm, tar_norm.T)
        return similarity
    else:
        qry = torch.tensor(qry, dtype=torch.float32).cuda()
        tar = torch.tensor(tar, dtype=torch.float32).cuda()
        qry_norm = torch.nn.functional.normalize(qry, p=2, dim=-1)
        tar_norm = torch.nn.functional.normalize(tar, p=2, dim=-1)

        # 计算归一化后向量的点积得到余弦相似度
        similarity = torch.matmul(qry_norm, tar_norm.T)
        return similarity

def get_iid2tag(benchmarkpath):
    bench = get_benchmark_info(benchmarkpath)
    iid2tag = {}
    for each in bench:
        iid2tag[each['item_id']] = each['label']
    return iid2tag

def image_to_base64(image_path):
    image_type = os.path.basename(image_path).split('.')[1]
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    input_base64 = f"data:image/{image_type};base64," + encoded_string
    #print(image_type)
    return input_base64

def get_benchmark_info(json_path):
# 读取jsonl文件内容到列表
    data_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line))
    
    return data_list

def get_embedding(input_item):
    """调用火山引擎API获取单个文本或图片的向量表示"""
    print('调用火山引擎API')
    client = Ark(api_key='5d9f219a-c2e0-4c5e-81de-19ec4d78cfe8')
    
    try:
        resp = client.multimodal_embeddings.create(
            model="ep-20250701102437-c8npn",
            input = input_item
        )
        if hasattr(resp, 'data') and isinstance(resp.data, dict) and 'embedding' in resp.data:
            embedding = resp.data['embedding']
            # 确保向量是numpy数组并展平为一维
            embedding = np.array(embedding).flatten()
            return embedding
            print('embedding生成完成')
        else:
            raise ValueError("API响应格式不符合预期，无法获取嵌入向量")
    except Exception as e:
        print(f" 获取向量失败, 错误: {str(e)}")
        return None

def get_embedding_from_raw(item_id, qry_text, qry_image_path):
    # emb_root_dir = '/mnt/bn/llm-data-chengyuhan-video/distillation/data/doubao_embeddings_for_distillation'
    emb_root_dir = '/mnt/bn/llm-data-chengyuhan-video/data/classification/doubao_16_all_embedding_v1description'
    emb_path = os.path.join(emb_root_dir, item_id+'.npy')
    #print("debug: processing embedding:", emb_path)
    # if os.path.exists(emb_path):
    #     return np.load(emb_path)
    # else:
        #return None
    input_item = []
    if qry_text:
        input_item.append({"type": "text", "text": qry_text})
    if qry_image_path:
        for img_path in qry_image_path:
            img_item = image_to_base64(img_path)
            input_item.append({"type": "image_url", "image_url": {"url": img_item}})
    emb = get_embedding(input_item)
    if emb is None:
        print(f"item_id: {item_id} embedding is None, skip")
        return None
    np.save(emb_path, emb)
    return emb


