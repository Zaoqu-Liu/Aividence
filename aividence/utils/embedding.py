"""
Embedding and similarity search utilities
"""

import numpy as np
import faiss
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def find_relevant_papers(
    abstracts: List[str], 
    claim: str, 
    embedding_model: SentenceTransformer,
    max_papers: int = 30
) -> Tuple[List[int], List[float]]:
    """
    Find papers most relevant to the claim using embedding similarity.
    
    Args:
        abstracts: List of paper abstracts
        claim: Scientific claim
        embedding_model: SentenceTransformer model
        max_papers: Maximum number of papers to return
        
    Returns:
        Tuple of (indices, distances)
    """
    if not abstracts:
        return [], []
        
    print(f"Encoding {len(abstracts)} abstracts...")
    
    # 我们不需要修改这部分，因为 SentenceTransformer 的 encode 方法
    # 已经通过 show_progress_bar=True 参数启用了进度条
    corpus_embeddings = embedding_model.encode(
        abstracts, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print("Encoding claim...")
    # 对于短文本（声明），不需要进度条
    claim_embedding = embedding_model.encode([claim], convert_to_numpy=True)
    
    # 使用 tqdm 为构建索引添加进度条
    print("Building search index...")
    dimension = corpus_embeddings.shape[1]
    
    with tqdm(total=1, desc="Building FAISS index") as pbar:
        index = faiss.IndexFlatL2(dimension)
        index.add(corpus_embeddings.astype('float32'))
        pbar.update(1)
    
    # 查找最相似的论文摘要
    k = min(max_papers, len(abstracts))
    print(f"Finding top {k} most relevant papers...")
    
    with tqdm(total=1, desc="Searching") as pbar:
        D, I = index.search(claim_embedding.astype('float32'), k=k)
        pbar.update(1)
    
    # 输出关于找到的最相关论文的一些信息
    print(f"Found {len(I[0])} relevant papers for detailed analysis")
    
    return I[0], D[0]