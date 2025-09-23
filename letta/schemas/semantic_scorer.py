"""
Semantic Scorer for MoE Role Memory System

This module provides integration with LLM services for semantic scoring of memory content.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import hashlib
import time

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """LLM响应模型"""
    content: str = Field(..., description="LLM返回的内容")
    usage: Optional[Dict[str, Any]] = Field(None, description="使用统计")
    model: Optional[str] = Field(None, description="使用的模型")
    finish_reason: Optional[str] = Field(None, description="完成原因")


class SemanticScorer(ABC):
    """语义评分器抽象基类"""
    
    @abstractmethod
    async def score_roles(
        self, 
        memory_content: str, 
        context: str, 
        role_descriptions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        为角色打分
        
        Args:
            memory_content: 记忆内容
            context: 上下文
            role_descriptions: 角色描述列表
            
        Returns:
            Dict[str, float]: 角色ID到分数的映射
        """
        pass


class OpenAISemanticScorer(SemanticScorer):
    """基于OpenAI的语义评分器"""
    
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        timeout: int = 10
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        """获取OpenAI客户端"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError("Please install openai package: pip install openai")
        return self._client
    
    async def score_roles(
        self, 
        memory_content: str, 
        context: str, 
        role_descriptions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """使用OpenAI进行语义评分"""
        
        # 构建提示词
        prompt = self._build_scoring_prompt(memory_content, context, role_descriptions)
        
        try:
            client = self._get_client()
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个智能记忆路由系统，专门负责分析内容语义并为角色打分。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 降低随机性，提高一致性
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            scores = self._parse_llm_response(content, role_descriptions)
            
            return scores
            
        except Exception as e:
            # 记录错误并返回空结果
            import logging
            logging.error(f"OpenAI semantic scoring failed: {e}")
            return {}
    
    def _build_scoring_prompt(
        self, 
        memory_content: str, 
        context: str, 
        role_descriptions: List[Dict[str, Any]]
    ) -> str:
        """构建语义评分提示词"""
        
        # 构建角色信息
        roles_info = ""
        for i, role in enumerate(role_descriptions, 1):
            roles_info += f"""{i}. {role['name']} ({role['id']})
   类型: {role['type']}
   描述: {role['description']}
   关键词: {', '.join(role['keywords']) if role['keywords'] else '无'}

"""
        
        prompt = f"""你是一个智能记忆路由系统。请根据记忆内容的语义含义，为每个角色打分（0-1分）。

记忆内容: "{memory_content}"
上下文: "{context}"

可用角色:
{roles_info}

请仔细分析记忆内容的语义，考虑以下因素:
1. 内容主题与角色职责的匹配度
2. 内容的专业领域与角色专长的相关性  
3. 内容的情感色彩与角色特征的契合度
4. 上下文信息对角色选择的影响
5. 内容的复杂度与角色处理能力的匹配

评分标准:
- 1.0: 完全相关，该角色是处理此内容的最佳选择
- 0.8-0.9: 高度相关，该角色非常适合处理此内容
- 0.6-0.7: 中度相关，该角色可以处理此内容
- 0.3-0.5: 低度相关，该角色勉强可以处理此内容
- 0.0-0.2: 基本无关，该角色不适合处理此内容

请以JSON格式返回评分结果，格式如下:
{{
    "{role_descriptions[0]['id'] if role_descriptions else 'role_id'}": 0.8,
    "角色ID2": 0.6,
    ...
}}

注意:
- 评分范围严格为0-1的浮点数
- 可以给多个角色打高分，如果内容与多个角色都相关
- 请基于深度语义理解，不要仅仅依赖关键词匹配
- 考虑内容的细微差别和上下文含义
- 只返回JSON格式的评分，不要包含其他解释文字"""
        
        return prompt
    
    def _parse_llm_response(self, content: str, role_descriptions: List[Dict[str, Any]]) -> Dict[str, float]:
        """解析LLM响应，提取评分"""
        try:
            # 尝试直接解析JSON
            scores = json.loads(content.strip())
            
            # 验证和清理分数
            cleaned_scores = {}
            valid_role_ids = {role['id'] for role in role_descriptions}
            
            for role_id, score in scores.items():
                if role_id in valid_role_ids:
                    # 确保分数在0-1范围内
                    score = max(0.0, min(1.0, float(score)))
                    cleaned_scores[role_id] = score
            
            return cleaned_scores
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # JSON解析失败，尝试从文本中提取
            import logging
            logging.warning(f"Failed to parse LLM response as JSON: {e}")
            logging.warning(f"LLM response content: {content}")
            
            return self._extract_scores_from_text(content, role_descriptions)
    
    def _extract_scores_from_text(self, content: str, role_descriptions: List[Dict[str, Any]]) -> Dict[str, float]:
        """从文本中提取评分（当JSON解析失败时）"""
        import re
        
        scores = {}
        valid_role_ids = {role['id'] for role in role_descriptions}
        
        # 尝试匹配各种可能的格式
        patterns = [
            r'"(\w+)":\s*([0-9]*\.?[0-9]+)',  # "role_id": 0.8
            r'(\w+):\s*([0-9]*\.?[0-9]+)',    # role_id: 0.8
            r'(\w+)\s*=\s*([0-9]*\.?[0-9]+)', # role_id = 0.8
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for role_id, score_str in matches:
                if role_id in valid_role_ids:
                    try:
                        score = max(0.0, min(1.0, float(score_str)))
                        scores[role_id] = score
                    except ValueError:
                        continue
        
        return scores


class AnthropicSemanticScorer(SemanticScorer):
    """基于Anthropic Claude的语义评分器"""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307", timeout: int = 10):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        """获取Anthropic客户端"""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError("Please install anthropic package: pip install anthropic")
        return self._client
    
    async def score_roles(
        self, 
        memory_content: str, 
        context: str, 
        role_descriptions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """使用Claude进行语义评分"""
        
        # 构建提示词（与OpenAI类似）
        prompt = self._build_scoring_prompt(memory_content, context, role_descriptions)
        
        try:
            client = self._get_client()
            
            response = await client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            scores = self._parse_llm_response(content, role_descriptions)
            
            return scores
            
        except Exception as e:
            import logging
            logging.error(f"Anthropic semantic scoring failed: {e}")
            return {}
    
    def _build_scoring_prompt(self, memory_content: str, context: str, role_descriptions: List[Dict[str, Any]]) -> str:
        """构建语义评分提示词（复用OpenAI的实现）"""
        scorer = OpenAISemanticScorer()
        return scorer._build_scoring_prompt(memory_content, context, role_descriptions)
    
    def _parse_llm_response(self, content: str, role_descriptions: List[Dict[str, Any]]) -> Dict[str, float]:
        """解析LLM响应（复用OpenAI的实现）"""
        scorer = OpenAISemanticScorer()
        return scorer._parse_llm_response(content, role_descriptions)
    
    def _extract_scores_from_text(self, content: str, role_descriptions: List[Dict[str, Any]]) -> Dict[str, float]:
        """从文本中提取评分（复用OpenAI的实现）"""
        scorer = OpenAISemanticScorer()
        return scorer._extract_scores_from_text(content, role_descriptions)


class MockSemanticScorer(SemanticScorer):
    """模拟语义评分器（用于测试和开发）"""
    
    def __init__(self, response_delay: float = 0.1):
        self.response_delay = response_delay
    
    async def score_roles(
        self, 
        memory_content: str, 
        context: str, 
        role_descriptions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """模拟语义评分"""
        
        # 模拟网络延迟
        await asyncio.sleep(self.response_delay)
        
        scores = {}
        content = f"{memory_content} {context}".lower()
        
        # 基于简单规则的模拟评分
        for role in role_descriptions:
            role_id = role['id']
            role_type = role.get('type', '').lower()
            role_keywords = [kw.lower() for kw in role.get('keywords', [])]
            
            score = 0.1  # 基础分数
            
            # 基于角色类型的启发式评分
            if role_type == 'technical' and any(word in content for word in ['代码', 'bug', '程序', '技术', '开发', 'code', 'program']):
                score += 0.7
            elif role_type == 'professional' and any(word in content for word in ['工作', '项目', '会议', '任务', 'work', 'project', 'meeting']):
                score += 0.6
            elif role_type == 'personal' and any(word in content for word in ['个人', '生活', '家庭', '朋友', 'personal', 'family', 'life']):
                score += 0.5
            elif role_type == 'health' and any(word in content for word in ['健康', '运动', '饮食', '锻炼', 'health', 'exercise', 'diet']):
                score += 0.6
            
            # 基于关键词的精确匹配
            for keyword in role_keywords:
                if keyword in content:
                    score += 0.2
            
            # 限制分数范围
            score = max(0.0, min(1.0, score))
            scores[role_id] = score
        
        return scores


def create_semantic_scorer(
    scorer_type: str = "mock",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> SemanticScorer:
    """
    创建语义评分器
    
    Args:
        scorer_type: 评分器类型 ("openai", "anthropic", "mock")
        api_key: API密钥
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        SemanticScorer: 语义评分器实例
    """
    
    if scorer_type.lower() == "openai":
        return OpenAISemanticScorer(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            **kwargs
        )
    elif scorer_type.lower() == "anthropic":
        return AnthropicSemanticScorer(
            api_key=api_key,
            model=model or "claude-3-haiku-20240307",
            **kwargs
        )
    elif scorer_type.lower() == "mock":
        return MockSemanticScorer(**kwargs)
    else:
        raise ValueError(f"Unsupported scorer type: {scorer_type}")
