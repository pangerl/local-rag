#!/usr/bin/env python3
"""
JiebaChunker 演示脚本

演示中文文本的分词效果和语义连贯性
"""

from app.core.chunker import JiebaChunker


def main():
    """主函数"""
    chunker = JiebaChunker()
    
    # 测试文本
    test_text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
    它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
    自从1956年达特茅斯会议提出人工智能概念以来，经过60多年的发展，
    人工智能已经从理论走向实践，从实验室走向产业化应用。
    近年来，随着深度学习技术的突破，人工智能迎来了新的发展机遇。
    """
    
    print("=" * 60)
    print("JiebaChunker 中文文本分片演示")
    print("=" * 60)
    
    print(f"\n原始文本：\n{test_text.strip()}")
    
    # 测试分词效果
    print(f"\n{'='*20} 分词效果 {'='*20}")
    tokens = chunker._tokenize(test_text)
    print(f"词元总数: {len(tokens)}")
    print(f"前20个词元: {tokens[:20]}")
    
    # 测试不同参数的分片效果
    test_cases = [
        (10, 2, "小分片，小重叠"),
        (15, 3, "中等分片，小重叠"),
        (20, 5, "大分片，中等重叠"),
    ]
    
    for chunk_size, chunk_overlap, description in test_cases:
        print(f"\n{'='*20} {description} {'='*20}")
        print(f"参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        chunks = chunker.chunk_text(test_text, chunk_size, chunk_overlap)
        
        print(f"分片数量: {len(chunks)}")
        
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = chunker._tokenize(chunk)
            print(f"\n分片 {i} (词元数: {len(chunk_tokens)}):")
            print(f"内容: {chunk}")
    
    # 测试中英文混合文本
    print(f"\n{'='*20} 中英文混合文本测试 {'='*20}")
    mixed_text = """
    Python是一种高级编程语言，由Guido van Rossum在1989年发明。
    它具有简洁的语法和强大的功能，广泛应用于Web开发、数据科学、
    机器学习等领域。TensorFlow和PyTorch是两个流行的深度学习框架。
    """
    
    print(f"原始文本: {mixed_text.strip()}")
    
    chunks = chunker.chunk_text(mixed_text, 12, 3)
    print(f"分片数量: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        chunk_tokens = chunker._tokenize(chunk)
        print(f"\n分片 {i} (词元数: {len(chunk_tokens)}):")
        print(f"内容: {chunk}")


if __name__ == "__main__":
    main()