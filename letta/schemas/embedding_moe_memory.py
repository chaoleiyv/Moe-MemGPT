"""
Embedding-based MoE (Mixture of Experts) Memory System for Letta

完全基于embedding的MoE角色记忆系统，模仿Archival Memory的实现方式。
所有角色描述和记忆内容都通过向量化进行语义匹配和路由。
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
from enum import Enum
from io import StringIO
from typing import Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, field_validator

from letta.constants import CORE_MEMORY_BLOCK_CHAR_LIMIT, MAX_EMBEDDING_DIM
from letta.llm_api.llm_client import LLMClient
from letta.otel.tracing import trace_method
from letta.schemas.block import Block, FileBlock
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import AgentType
from letta.schemas.letta_base import LettaBase
from letta.schemas.passage import Passage

logger = logging.getLogger(__name__)


class RoleType(str, Enum):
    """预定义的角色类型"""
    PROFESSIONAL = "professional"      # 专业/工作相关
    PERSONAL = "personal"             # 个人生活相关
    TECHNICAL = "technical"           # 技术相关
    CREATIVE = "creative"             # 创意相关
    SOCIAL = "social"                 # 社交相关
    ACADEMIC = "academic"             # 学术相关
    HEALTH = "health"                 # 健康相关
    FINANCE = "finance"               # 财务相关
    TRAVEL = "travel"                 # 旅行相关
    GENERAL = "general"               # 通用角色（替代原来的全局记忆）
    CUSTOM = "custom"                 # 自定义角色


class EmbeddingRoleConfig(BaseModel):
    """基于Embedding的角色配置"""
    role_id: str = Field(..., description="角色唯一标识")
    role_type: RoleType = Field(default=RoleType.CUSTOM, description="角色类型")
    role_name: str = Field(..., description="角色名称")
    
    # 角色描述和向量化
    role_description: str = Field(..., description="角色详细描述，用于生成embedding")
    role_embedding: Optional[List[float]] = Field(None, description="角色描述的向量表示")
    
    # 关键词作为辅助（保留原有功能）
    keywords: List[str] = Field(default_factory=list, description="角色关键词（辅助匹配）")
    
    # 记忆管理配置
    max_memory_size: Optional[int] = Field(default=50, description="最大记忆块数量")
    memory_retention_strategy: str = Field(default="semantic", description="记忆保留策略：semantic, priority, fifo")
    
    # 激活阈值
    activation_threshold: float = Field(default=0.3, description="角色激活的语义相似度阈值")
    
    def __init__(self, **data):
        super().__init__(**data)
        # 如果没有提供embedding，需要异步生成
        if not self.role_embedding:
            logger.info(f"Role {self.role_id} needs embedding generation for description: {self.role_description[:100]}...")


class EmbeddingRoleMemoryBlock(Block):
    """基于Embedding的角色记忆块"""
    role_id: str = Field(..., description="所属角色ID")
    
    # 向量化字段（模仿Passage的结构）
    embedding: Optional[List[float]] = Field(None, description="记忆内容的向量表示")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="用于生成embedding的配置")
    
    # 语义相关性评分
    semantic_score: float = Field(default=0.0, description="与角色的语义相关性评分")
    
    # 访问统计
    access_count: int = Field(default=0, description="访问次数")
    last_accessed: datetime = Field(default_factory=datetime.utcnow, description="最后访问时间")
    
    @field_validator("embedding")
    @classmethod
    def pad_embeddings(cls, embedding: Optional[List[float]]) -> Optional[List[float]]:
        """填充embedding到标准维度（模仿Passage的验证器）"""
        if embedding and len(embedding) != MAX_EMBEDDING_DIM:
            import numpy as np
            np_embedding = np.array(embedding)
            padded_embedding = np.pad(np_embedding, (0, MAX_EMBEDDING_DIM - np_embedding.shape[0]), mode="constant")
            return padded_embedding.tolist()
        return embedding
    
    def update_access_info(self):
        """更新访问信息"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class EmbeddingRoleMemoryRepository(BaseModel):
    """基于Embedding的单个角色记忆仓库"""
    config: EmbeddingRoleConfig = Field(..., description="角色配置")
    core_memory_blocks: List[EmbeddingRoleMemoryBlock] = Field(default_factory=list, description="核心记忆块")
    archival_passages: List[Passage] = Field(default_factory=list, description="档案记忆段落（完全使用Passage结构）")
    file_blocks: List[FileBlock] = Field(default_factory=list, description="文件记忆块")
    
    async def add_memory_block_async(self, block: EmbeddingRoleMemoryBlock, embedding_config: EmbeddingConfig, actor) -> bool:
        """异步添加记忆块，自动生成embedding"""
        if self.config.max_memory_size and len(self.core_memory_blocks) >= self.config.max_memory_size:
            await self._apply_retention_strategy_async(embedding_config, actor)
        
        # 为记忆块生成embedding（如果还没有）
        if not block.embedding and block.value:
            block.embedding = await self._generate_memory_embedding(block.value, embedding_config, actor)
            block.embedding_config = embedding_config
        
        # 计算与角色的语义相关性
        if block.embedding and self.config.role_embedding:
            block.semantic_score = self._calculate_semantic_similarity(block.embedding, self.config.role_embedding)
        
        self.core_memory_blocks.append(block)
        return True
    
    async def add_archival_passage_async(self, text: str, embedding_config: EmbeddingConfig, actor, tags: Optional[List[str]] = None) -> Passage:
        """添加档案记忆段落，完全模仿Archival Memory的实现"""
        # 生成embedding
        embedding = await self._generate_memory_embedding(text, embedding_config, actor)
        
        # 创建Passage对象（完全使用原有结构）
        passage = Passage(
            text=text,
            embedding=embedding,
            embedding_config=embedding_config,
            tags=tags or [],
            created_at=datetime.utcnow()
        )
        
        self.archival_passages.append(passage)
        return passage
    
    async def _generate_memory_embedding(self, text: str, embedding_config: EmbeddingConfig, actor) -> List[float]:
        """生成记忆内容的embedding（模仿PassageManager的实现）"""
        embedding_client = LLMClient.create(
            provider_type=embedding_config.embedding_endpoint_type,
            actor=actor,
        )
        
        embeddings = await embedding_client.request_embeddings([text], embedding_config)
        return embeddings[0] if embeddings else []
    
    def _calculate_semantic_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """计算两个embedding的余弦相似度"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    async def _apply_retention_strategy_async(self, embedding_config: EmbeddingConfig, actor):
        """应用记忆保留策略"""
        if self.config.memory_retention_strategy == "semantic":
            # 基于语义相似度的保留策略：移除与角色最不相关的记忆
            if self.core_memory_blocks:
                self.core_memory_blocks.sort(key=lambda x: x.semantic_score)
                removed_block = self.core_memory_blocks.pop(0)
                logger.info(f"Removed memory block with semantic score {removed_block.semantic_score}")
        elif self.config.memory_retention_strategy == "priority":
            # 基于优先级的保留策略
            if self.core_memory_blocks:
                self.core_memory_blocks.sort(key=lambda x: (x.semantic_score, x.access_count))
                self.core_memory_blocks.pop(0)
        elif self.config.memory_retention_strategy == "fifo":
            # 先进先出
            if self.core_memory_blocks:
                self.core_memory_blocks.pop(0)
    
    async def search_memories_async(self, query: str, embedding_config: EmbeddingConfig, actor, top_k: int = 5) -> List[EmbeddingRoleMemoryBlock]:
        """基于embedding的语义搜索（模仿archival_memory_search）"""
        if not self.core_memory_blocks:
            return []
        
        # 生成查询的embedding
        query_embedding = await self._generate_memory_embedding(query, embedding_config, actor)
        
        # 计算与所有记忆块的相似度
        scored_blocks = []
        for block in self.core_memory_blocks:
            if block.embedding:
                similarity = self._calculate_semantic_similarity(query_embedding, block.embedding)
                scored_blocks.append((similarity, block))
        
        # 按相似度排序并返回top_k
        scored_blocks.sort(key=lambda x: x[0], reverse=True)
        
        # 更新访问信息
        result_blocks = []
        for similarity, block in scored_blocks[:top_k]:
            block.update_access_info()
            result_blocks.append(block)
        
        return result_blocks
    
    async def search_archival_memories_async(self, query: str, embedding_config: EmbeddingConfig, actor, top_k: int = 5) -> List[Passage]:
        """搜索档案记忆（完全模仿archival_memory_search）"""
        if not self.archival_passages:
            return []
        
        # 生成查询的embedding
        query_embedding = await self._generate_memory_embedding(query, embedding_config, actor)
        
        # 计算与所有档案段落的相似度
        scored_passages = []
        for passage in self.archival_passages:
            if passage.embedding:
                similarity = self._calculate_semantic_similarity(query_embedding, passage.embedding)
                scored_passages.append((similarity, passage))
        
        # 按相似度排序并返回top_k
        scored_passages.sort(key=lambda x: x[0], reverse=True)
        return [passage for similarity, passage in scored_passages[:top_k]]


class EmbeddingMoEMemory(BaseModel, validate_assignment=True):
    """
    基于Embedding的纯MoE记忆系统
    
    完全基于向量语义匹配的角色记忆管理，模仿Archival Memory的实现方式
    """
    
    agent_type: Optional[Union[AgentType, str]] = Field(None, description="Agent type controlling prompt rendering.")
    
    # 角色仓库
    role_repositories: Dict[str, EmbeddingRoleMemoryRepository] = Field(
        default_factory=dict, 
        description="角色记忆仓库字典，key为role_id"
    )
    
    # Embedding配置
    embedding_config: EmbeddingConfig = Field(..., description="Embedding配置")
    
    # MoE参数
    routing_threshold: float = Field(default=0.3, description="路由阈值")
    max_active_roles: int = Field(default=3, description="同时激活的最大角色数")
    
    # 当前状态
    _current_active_roles: List[str] = Field(default_factory=list, description="当前激活的角色列表")
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.role_repositories:
            asyncio.create_task(self._create_default_roles_async())
    
    async def _create_default_roles_async(self):
        """创建默认角色并生成embeddings"""
        default_roles = [
            {
                "role_id": "professional_assistant",
                "role_name": "专业助手",
                "role_description": "专业工作助手，擅长处理商务沟通、项目管理、技术咨询、工作流程优化等专业领域的任务。具有严谨的工作态度和专业的沟通方式。",
                "keywords": ["工作", "项目", "业务", "专业", "技术", "管理", "咨询"]
            },
            {
                "role_id": "personal_assistant", 
                "role_name": "个人助手",
                "role_description": "贴心的个人生活助手，帮助处理日常生活事务、健康管理、娱乐休闲、人际关系等个人生活相关的需求。语言亲切自然。",
                "keywords": ["生活", "个人", "健康", "娱乐", "休闲", "家庭", "朋友"]
            },
            {
                "role_id": "technical_expert",
                "role_name": "技术专家", 
                "role_description": "资深技术专家，精通编程开发、系统架构、技术调研、问题排查等技术领域。能提供深入的技术分析和解决方案。",
                "keywords": ["编程", "开发", "技术", "代码", "架构", "算法", "系统"]
            }
        ]
        
        # 注意：在实际使用中，需要传入actor来生成embeddings
        # 这里只创建配置，embedding将在首次使用时生成
        for role_data in default_roles:
            config = EmbeddingRoleConfig(**role_data)
            repository = EmbeddingRoleMemoryRepository(config=config)
            self.role_repositories[role_data["role_id"]] = repository
    
    @trace_method
    async def route_memory_to_role_async(self, memory_content: str, context: str, actor, use_semantic_scoring: bool = True) -> List[str]:
        """
        基于embedding相似度将记忆路由到合适的角色库
        完全基于语义匹配，不使用关键词
        """
        if not self.role_repositories:
            return []
        
        # 生成内容的embedding
        embedding_client = LLMClient.create(
            provider_type=self.embedding_config.embedding_endpoint_type,
            actor=actor,
        )
        
        content_embeddings = await embedding_client.request_embeddings([memory_content], self.embedding_config)
        content_embedding = content_embeddings[0] if content_embeddings else []
        
        if not content_embedding:
            logger.warning("Failed to generate embedding for memory content")
            return []
        
        # 确保所有角色都有embedding
        await self._ensure_role_embeddings_async(actor)
        
        # 计算与每个角色的相似度
        role_scores = {}
        for role_id, repository in self.role_repositories.items():
            if repository.config.role_embedding:
                similarity = repository._calculate_semantic_similarity(content_embedding, repository.config.role_embedding)
                role_scores[role_id] = similarity
        
        # 选择相似度超过阈值的角色
        selected_roles = []
        for role_id, score in role_scores.items():
            if score >= self.routing_threshold:
                selected_roles.append(role_id)
                logger.info(f"Memory routed to role {role_id} with similarity score {score:.3f}")
        
        # 如果没有角色超过阈值，选择相似度最高的角色
        if not selected_roles and role_scores:
            best_role = max(role_scores.items(), key=lambda x: x[1])
            selected_roles.append(best_role[0])
            logger.info(f"Memory routed to best matching role {best_role[0]} with similarity score {best_role[1]:.3f}")
        
        return selected_roles
    
    @trace_method
    async def moe_gate_for_context_async(self, context: str, actor, max_roles: Optional[int] = None) -> List[str]:
        """
        基于embedding的MoE门控机制，选择与上下文最相关的角色
        """
        if not self.role_repositories:
            return []
        
        max_roles = max_roles or self.max_active_roles
        
        # 生成上下文的embedding
        embedding_client = LLMClient.create(
            provider_type=self.embedding_config.embedding_endpoint_type,
            actor=actor,
        )
        
        context_embeddings = await embedding_client.request_embeddings([context], self.embedding_config)
        context_embedding = context_embeddings[0] if context_embeddings else []
        
        if not context_embedding:
            logger.warning("Failed to generate embedding for context")
            return list(self.role_repositories.keys())[:max_roles]
        
        # 确保所有角色都有embedding
        await self._ensure_role_embeddings_async(actor)
        
        # 计算与每个角色的相似度
        role_scores = []
        for role_id, repository in self.role_repositories.items():
            if repository.config.role_embedding:
                similarity = repository._calculate_semantic_similarity(context_embedding, repository.config.role_embedding)
                role_scores.append((similarity, role_id))
        
        # 按相似度排序，选择top_k个角色
        role_scores.sort(key=lambda x: x[0], reverse=True)
        active_roles = [role_id for similarity, role_id in role_scores[:max_roles] if similarity >= self.routing_threshold]
        
        # 如果没有角色超过阈值，至少激活相似度最高的角色
        if not active_roles and role_scores:
            active_roles = [role_scores[0][1]]
        
        self._current_active_roles = active_roles
        
        logger.info(f"Activated roles for context: {active_roles}")
        for similarity, role_id in role_scores[:max_roles]:
            logger.debug(f"Role {role_id}: similarity {similarity:.3f}")
        
        return active_roles
    
    async def _ensure_role_embeddings_async(self, actor):
        """确保所有角色都有embedding"""
        embedding_client = LLMClient.create(
            provider_type=self.embedding_config.embedding_endpoint_type,
            actor=actor,
        )
        
        for role_id, repository in self.role_repositories.items():
            if not repository.config.role_embedding:
                logger.info(f"Generating embedding for role {role_id}")
                embeddings = await embedding_client.request_embeddings([repository.config.role_description], self.embedding_config)
                repository.config.role_embedding = embeddings[0] if embeddings else []
    
    @trace_method
    async def smart_add_memory_async(self, memory_content: str, context: str, actor, tags: Optional[List[str]] = None) -> bool:
        """智能添加记忆到合适的角色库"""
        # 路由到合适的角色
        target_roles = await self.route_memory_to_role_async(memory_content, context, actor)
        
        if not target_roles:
            logger.warning("No suitable role found for memory content")
            return False
        
        # 创建记忆块并添加到目标角色库
        success = True
        for role_id in target_roles:
            if role_id in self.role_repositories:
                memory_block = EmbeddingRoleMemoryBlock(
                    role_id=role_id,
                    label=f"memory_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    value=memory_content,
                    tags=tags or []
                )
                
                try:
                    await self.role_repositories[role_id].add_memory_block_async(memory_block, self.embedding_config, actor)
                    logger.info(f"Successfully added memory to role {role_id}")
                except Exception as e:
                    logger.error(f"Failed to add memory to role {role_id}: {e}")
                    success = False
        
        return success
    
    @trace_method
    async def search_role_memories_async(self, query: str, actor, role_ids: Optional[List[str]] = None, top_k: int = 5) -> Dict[str, List[EmbeddingRoleMemoryBlock]]:
        """搜索指定角色的记忆"""
        if role_ids is None:
            role_ids = self._current_active_roles or list(self.role_repositories.keys())
        
        results = {}
        for role_id in role_ids:
            if role_id in self.role_repositories:
                try:
                    role_results = await self.role_repositories[role_id].search_memories_async(query, self.embedding_config, actor, top_k)
                    if role_results:
                        results[role_id] = role_results
                except Exception as e:
                    logger.error(f"Error searching memories in role {role_id}: {e}")
        
        return results
    
    @trace_method
    def compile(self, tool_usage_rules=None, sources=None, max_files_open=None) -> str:
        """渲染激活角色的记忆到prompt中"""
        s = StringIO()
        
        if not self._current_active_roles:
            s.write("<memory_blocks>\nNo active role memories.\n</memory_blocks>")
            return s.getvalue()
        
        s.write("<memory_blocks>\n以下是当前激活角色的记忆块：\n\n")
        
        for role_id in self._current_active_roles:
            if role_id in self.role_repositories:
                repository = self.role_repositories[role_id]
                role_name = repository.config.role_name
                
                s.write(f"<role id=\"{role_id}\" name=\"{role_name}\">\n")
                
                # 渲染角色的核心记忆块
                if repository.core_memory_blocks:
                    for idx, block in enumerate(repository.core_memory_blocks[:5]):  # 限制显示数量
                        s.write(f"<memory_block>\n")
                        s.write(f"<label>{block.label}</label>\n")
                        s.write(f"<content>{block.value}</content>\n")
                        s.write(f"<semantic_score>{block.semantic_score:.3f}</semantic_score>\n")
                        s.write(f"</memory_block>\n")
                        if idx < min(4, len(repository.core_memory_blocks) - 1):
                            s.write("\n")
                
                s.write(f"</role>\n")
                if role_id != self._current_active_roles[-1]:
                    s.write("\n")
        
        s.write("\n</memory_blocks>")
        
        # 处理其他组件（工具规则、文件等）
        if tool_usage_rules is not None:
            desc = getattr(tool_usage_rules, "description", None) or ""
            val = getattr(tool_usage_rules, "value", None) or ""
            s.write("\n\n<tool_usage_rules>\n")
            s.write(f"{desc}\n\n")
            s.write(f"{val}\n")
            s.write("</tool_usage_rules>")
        
        return s.getvalue()
    
    def get_memory_statistics(self) -> Dict:
        """获取记忆统计信息"""
        stats = {
            "total_roles": len(self.role_repositories),
            "active_roles": len(self._current_active_roles),
            "total_memory_blocks": 0,
            "total_archival_passages": 0,
            "role_details": {}
        }
        
        for role_id, repository in self.role_repositories.items():
            role_stats = {
                "role_name": repository.config.role_name,
                "memory_blocks": len(repository.core_memory_blocks),
                "archival_passages": len(repository.archival_passages),
                "is_active": role_id in self._current_active_roles,
                "has_embedding": repository.config.role_embedding is not None
            }
            stats["role_details"][role_id] = role_stats
            stats["total_memory_blocks"] += role_stats["memory_blocks"]
            stats["total_archival_passages"] += role_stats["archival_passages"]
        
        return stats


# 工厂函数
async def create_embedding_moe_memory_async(
    persona: str,
    human: str,
    agent_type: AgentType,
    embedding_config: EmbeddingConfig,
    actor
) -> EmbeddingMoEMemory:
    """创建基于embedding的MoE记忆系统"""
    
    # 创建记忆系统
    moe_memory = EmbeddingMoEMemory(
        agent_type=agent_type,
        embedding_config=embedding_config
    )
    
    # 等待默认角色创建完成
    await moe_memory._create_default_roles_async()
    
    # 确保所有角色都有embedding
    await moe_memory._ensure_role_embeddings_async(actor)
    
    # 添加初始的persona和human信息
    await moe_memory.smart_add_memory_async(
        memory_content=f"Persona: {persona}",
        context="initial setup",
        actor=actor,
        tags=["persona", "initial"]
    )
    
    await moe_memory.smart_add_memory_async(
        memory_content=f"Human: {human}",
        context="initial setup", 
        actor=actor,
        tags=["human", "initial"]
    )
    
    return moe_memory
