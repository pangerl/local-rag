# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import logging
from typing import List
import warnings

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_path: str, device: str = 'cpu', max_length: int = 8192):
        logger.info(f"Initializing EmbeddingModel from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        self.device = device
        self.max_length = max_length
        logger.info(f"EmbeddingModel initialized on device: {self.device}")

    @staticmethod
    def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> Tensor:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_dict = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch_dict.to(self.device)
            outputs = self.model(**batch_dict)
            embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)


class RerankerModel:
    def __init__(self, model_path: str, device: str = 'cpu', max_length: int = 8192):
        logger.info(f"Initializing RerankerModel from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        self.device = device
        self.max_length = max_length

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        logger.info(f"RerankerModel initialized on device: {self.device}")

    @torch.no_grad()
    def rerank(self, query: str, documents: List[str], instruction: str = None) -> List[float]:
        if not instruction:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'

        # 格式化查询-文档对
        formatted_pairs = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in documents]

        # 添加系统指令前缀和后缀
        full_texts = [self.prefix + pair + self.suffix for pair in formatted_pairs]

        # 使用 __call__ 方法进行高效处理
        inputs = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]

        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
