"""
向量检索服务模块
负责查询向量化、相似性搜索和结果重排序
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import ModelLoadError, DatabaseError


logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    向量检索器类
    
    负责查询向量化、相似性搜索和结果重排序，
    提供完整的检索功能和性能监控
    """
    
    def __init__(self, settings: Settings, db_service: ChromaDBService, 
                 model_loader: ModelLoader):
        """
        初始化向量检索器
        
        Args:
            settings: 系统配置对象
            db_service: ChromaDB 数据库服务
            model_loader: 模型加载器
        """
        self.settings = settings
        self.db_service = db_service
        self.model_loader = model_loader
        
        # 模型缓存
        self.embedding_model: Optional[SentenceTransformer] = None
        self.reranker_model: Optional[SentenceTransformer] = None
        
        # 检索统计
        self.retrieval_stats = {
            "total_queries": 0,
            "total_retrieval_time": 0.0,
            "total_rerank_time": 0.0,
            "average_retrieval_time": 0.0,
            "average_rerank_time": 0.0,
            "last_query_at": None
        }
        
        logger.info("向量检索器初始化完成")
    
    def _ensure_embedding_model(self) -> SentenceTransformer:
        """
        确保嵌入模型已加载
        
        Returns:
            SentenceTransformer: 嵌入模型实例
            
        Raises:
            ModelLoadError: 当模型加载失败时抛出异常
        """
        if self.embedding_model is None:
            try:
                logger.info("加载嵌入模型用于查询向量化")
                self.embedding_model = self.model_loader.load_embedding_model()
            except Exception as e:
                error_msg = f"嵌入模型加载失败: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg) from e
        
        return self.embedding_model
    
    def _ensure_reranker_model(self) -> SentenceTransformer:
        """
        确保重排序模型已加载
        
        Returns:
            SentenceTransformer: 重排序模型实例
            
        Raises:
            ModelLoadError: 当模型加载失败时抛出异常
        """
        if self.reranker_model is None:
            try:
                logger.info("加载重排序模型用于结果重排序")
                self.reranker_model = self.model_loader.load_reranker_model()
            except Exception as e:
                error_msg = f"重排序模型加载失败: {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg) from e
        
        return self.reranker_model
    
    def _vectorize_query(self, query: str) -> np.ndarray:
        """
        将查询文本向量化
        
        Args:
            query: 查询文本
            
        Returns:
            np.ndarray: 查询向量
            
        Raises:
            ModelLoadError: 当向量化失败时抛出异常
        """
        try:
            model = self._ensure_embedding_model()
            logger.debug(f"开始向量化查询: {query[:50]}...")
            
            # 生成查询向量
            query_vector = model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # 归一化向量，提高检索效果
            )
            
            logger.debug(f"查询向量化完成，维度: {query_vector.shape}")
            return query_vector[0]  # 返回单个向量
            
        except Exception as e:
            error_msg = f"查询向量化失败: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg) from e
    
    def _search_similar_chunks(self, query_vector: np.ndarray, 
                              retrieval_k: int) -> Dict[str, Any]:
        """
        搜索相似的文档分片
        
        Args:
            query_vector: 查询向量
            retrieval_k: 检索的候选文档数量
            
        Returns:
            Dict[str, Any]: 检索结果
            
        Raises:
            DatabaseError: 当数据库查询失败时抛出异常
        """
        try:
            logger.debug(f"开始相似性搜索，检索 {retrieval_k} 个候选结果")
            
            # 获取集合
            collection = self.db_service.create_or_get_collection()
            
            # 执行向量相似性搜索
            results = collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=retrieval_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 整理搜索结果
            if not results["ids"] or not results["ids"][0]:
                logger.info("未找到相似的文档分片")
                return {
                    "chunks": [],
                    "total_found": 0
                }
            
            chunks = []
            for i in range(len(results["ids"][0])):
                chunk = {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1.0 - results["distances"][0][i],  # 转换为相似度分数
                    "distance": results["distances"][0][i]
                }
                chunks.append(chunk)
            
            logger.debug(f"相似性搜索完成，找到 {len(chunks)} 个候选结果")
            
            return {
                "chunks": chunks,
                "total_found": len(chunks)
            }
            
        except Exception as e:
            error_msg = f"相似性搜索失败: {str(e)}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def _rerank_results(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用重排序模型对搜索结果进行重排序
        
        Args:
            query: 原始查询文本
            chunks: 候选文档分片列表
            
        Returns:
            List[Dict[str, Any]]: 重排序后的文档分片列表
            
        Raises:
            ModelLoadError: 当重排序失败时抛出异常
        """
        if not chunks:
            return chunks
        
        try:
            model = self._ensure_reranker_model()
            logger.debug(f"开始重排序 {len(chunks)} 个候选结果")
            
            # 准备查询-文档对
            query_doc_pairs = []
            for chunk in chunks:
                query_doc_pairs.append([query, chunk["text"]])
            
            # 计算重排序分数
            rerank_scores = model.predict(query_doc_pairs)
            
            # 如果返回的是数组，取第一列（相关性分数）
            if isinstance(rerank_scores, np.ndarray) and rerank_scores.ndim > 1:
                rerank_scores = rerank_scores[:, 0]
            
            # 添加重排序分数到结果中
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(rerank_scores[i])
            
            # 按重排序分数降序排序
            reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            
            logger.debug(f"重排序完成，结果已按相关性重新排序")
            
            return reranked_chunks
            
        except Exception as e:
            error_msg = f"结果重排序失败: {str(e)}"
            logger.error(error_msg)
            # 重排序失败时返回原始结果，不中断检索流程
            logger.warning("重排序失败，返回原始搜索结果")
            return chunks
    
    def retrieve(self, query: str, retrieval_k: Optional[int] = None, 
                top_k: Optional[int] = None, use_reranker: bool = True) -> Dict[str, Any]:
        """
        执行完整的检索流程
        
        Args:
            query: 查询文本
            retrieval_k: 候选文档数量，如果为 None 则使用默认值
            top_k: 返回结果数量，如果为 None 则使用默认值
            use_reranker: 是否使用重排序模型
            
        Returns:
            Dict[str, Any]: 检索结果
            
        Raises:
            ModelLoadError: 当模型操作失败时抛出异常
            DatabaseError: 当数据库操作失败时抛出异常
        """
        start_time = time.time()
        retrieval_k = retrieval_k or self.settings.DEFAULT_RETRIEVAL_K
        top_k = top_k or self.settings.DEFAULT_TOP_K
        
        try:
            logger.info(f"开始检索查询: {query[:100]}...")
            logger.info(f"检索参数: retrieval_k={retrieval_k}, top_k={top_k}, use_reranker={use_reranker}")
            
            # 1. 查询向量化
            query_vector = self._vectorize_query(query)
            
            # 2. 相似性搜索
            search_start = time.time()
            search_results = self._search_similar_chunks(query_vector, retrieval_k)
            search_time = time.time() - search_start
            
            chunks = search_results["chunks"]
            
            # 3. 重排序（如果启用）
            rerank_time = 0.0
            if use_reranker and chunks:
                rerank_start = time.time()
                chunks = self._rerank_results(query, chunks)
                rerank_time = time.time() - rerank_start
            
            # 4. 截取 top_k 结果
            final_chunks = chunks[:top_k] if chunks else []
            
            # 5. 计算总耗时
            total_time = time.time() - start_time
            
            # 6. 更新统计信息
            self._update_retrieval_stats(total_time, search_time, rerank_time)
            
            # 7. 构建返回结果
            result = {
                "query": query,
                "results": final_chunks,
                "total_candidates": search_results["total_found"],
                "returned_count": len(final_chunks),
                "retrieval_k": retrieval_k,
                "top_k": top_k,
                "use_reranker": use_reranker,
                "timing": {
                    "total_time": total_time,
                    "search_time": search_time,
                    "rerank_time": rerank_time
                }
            }
            
            logger.info(f"检索完成，返回 {len(final_chunks)} 个结果，"
                       f"总耗时: {total_time:.3f}s (搜索: {search_time:.3f}s, 重排序: {rerank_time:.3f}s)")
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"检索失败: {str(e)}, 耗时: {total_time:.3f}s"
            logger.error(error_msg)
            
            if isinstance(e, (ModelLoadError, DatabaseError)):
                raise
            else:
                raise DatabaseError(error_msg) from e
    
    def batch_retrieve(self, queries: List[str], retrieval_k: Optional[int] = None,
                      top_k: Optional[int] = None, use_reranker: bool = True) -> Dict[str, Any]:
        """
        批量检索多个查询
        
        Args:
            queries: 查询文本列表
            retrieval_k: 候选文档数量
            top_k: 返回结果数量
            use_reranker: 是否使用重排序模型
            
        Returns:
            Dict[str, Any]: 批量检索结果
        """
        start_time = time.time()
        
        logger.info(f"开始批量检索 {len(queries)} 个查询")
        
        results = {
            "total_queries": len(queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "query_results": [],
            "errors": []
        }
        
        for i, query in enumerate(queries):
            try:
                result = self.retrieve(query, retrieval_k, top_k, use_reranker)
                results["query_results"].append(result)
                results["successful_queries"] += 1
                
            except Exception as e:
                error_info = {
                    "query_index": i,
                    "query": query,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                results["errors"].append(error_info)
                results["failed_queries"] += 1
                
                logger.error(f"批量检索中查询失败: {query[:50]}..., 错误: {str(e)}")
        
        total_time = time.time() - start_time
        results["total_processing_time"] = total_time
        
        logger.info(f"批量检索完成，成功: {results['successful_queries']}, "
                   f"失败: {results['failed_queries']}, "
                   f"总耗时: {total_time:.2f}s")
        
        return results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        获取检索统计信息
        
        Returns:
            Dict[str, Any]: 检索统计信息
        """
        stats = self.retrieval_stats.copy()
        
        # 添加数据库统计信息
        try:
            db_stats = self.db_service.get_collection_info()
            stats.update({
                "total_documents_in_db": db_stats.get("count", 0),
                "collection_name": db_stats.get("name", "unknown")
            })
        except Exception as e:
            logger.warning(f"获取数据库统计信息失败: {str(e)}")
        
        return stats
    
    def _update_retrieval_stats(self, total_time: float, search_time: float, rerank_time: float):
        """
        更新检索统计信息
        
        Args:
            total_time: 总检索时间
            search_time: 搜索时间
            rerank_time: 重排序时间
        """
        self.retrieval_stats["total_queries"] += 1
        self.retrieval_stats["total_retrieval_time"] += search_time
        self.retrieval_stats["total_rerank_time"] += rerank_time
        
        # 计算平均时间
        total_queries = self.retrieval_stats["total_queries"]
        self.retrieval_stats["average_retrieval_time"] = (
            self.retrieval_stats["total_retrieval_time"] / total_queries
        )
        self.retrieval_stats["average_rerank_time"] = (
            self.retrieval_stats["total_rerank_time"] / total_queries
        )
        
        self.retrieval_stats["last_query_at"] = time.time()
        
        logger.debug(f"统计信息更新: 总查询数: {total_queries}, "
                    f"平均检索时间: {self.retrieval_stats['average_retrieval_time']:.3f}s")
    
    def health_check(self) -> Dict[str, Any]:
        """
        检索器健康检查
        
        Returns:
            Dict[str, Any]: 健康检查结果
        """
        health_info = {
            "status": "unknown",
            "components": {},
            "error": None
        }
        
        try:
            # 检查数据库服务
            db_health = self.db_service.health_check()
            health_info["components"]["database"] = db_health
            
            # 检查模型加载器
            model_info = self.model_loader.get_model_info()
            health_info["components"]["models"] = {
                "status": "healthy" if model_info["models_loaded"] else "unhealthy",
                "embedding_model_loaded": model_info["embedding_model_loaded"],
                "reranker_model_loaded": model_info["reranker_model_loaded"]
            }
            
            # 检查集合中是否有数据
            try:
                collection_info = self.db_service.get_collection_info()
                has_data = collection_info.get("count", 0) > 0
                health_info["components"]["data"] = {
                    "status": "healthy" if has_data else "warning",
                    "document_count": collection_info.get("count", 0),
                    "message": "有数据可供检索" if has_data else "数据库为空，无法进行检索"
                }
            except Exception as e:
                health_info["components"]["data"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            # 确定整体状态
            db_healthy = db_health["status"] == "healthy"
            models_healthy = health_info["components"]["models"]["status"] == "healthy"
            
            if db_healthy and models_healthy:
                health_info["status"] = "healthy"
            else:
                health_info["status"] = "unhealthy"
            
        except Exception as e:
            health_info["status"] = "error"
            health_info["error"] = str(e)
            logger.error(f"检索器健康检查失败: {str(e)}")
        
        return health_info
    
    def test_retrieval(self, test_query: str = "测试查询") -> Dict[str, Any]:
        """
        测试检索功能
        
        Args:
            test_query: 测试查询文本
            
        Returns:
            Dict[str, Any]: 测试结果
        """
        try:
            logger.info(f"开始测试检索功能，查询: {test_query}")
            
            result = self.retrieve(test_query, retrieval_k=5, top_k=3, use_reranker=True)
            
            test_result = {
                "status": "success",
                "test_query": test_query,
                "results_count": result["returned_count"],
                "total_candidates": result["total_candidates"],
                "timing": result["timing"],
                "message": f"检索测试成功，返回 {result['returned_count']} 个结果"
            }
            
            logger.info(f"检索功能测试完成: {test_result['message']}")
            return test_result
            
        except Exception as e:
            error_msg = f"检索功能测试失败: {str(e)}"
            logger.error(error_msg)
            
            return {
                "status": "failed",
                "test_query": test_query,
                "error": error_msg,
                "error_type": type(e).__name__
            }