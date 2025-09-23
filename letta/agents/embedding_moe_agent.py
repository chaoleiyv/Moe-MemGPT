"""
Embedding-based MoE Agent for Letta

基于embedding语义匹配的MoE角色智能体，完全使用向量化进行记忆管理和角色激活。
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

from letta.agent import LettaAgent
from letta.otel.tracing import trace_method
from letta.schemas.agent import AgentState, AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.embedding_moe_memory import EmbeddingMoEMemory, create_embedding_moe_memory_async
from letta.schemas.letta_response import LettaResponse
from letta.schemas.message import MessageCreate
from letta.schemas.role_templates import RoleTemplates

logger = logging.getLogger(__name__)


class EmbeddingMoEAgent(LettaAgent):
    """
    基于Embedding的纯MoE角色智能体
    
    使用向量语义匹配进行：
    1. 记忆路由：将新记忆分配到最相关的角色库
    2. 角色激活：根据上下文激活最相关的角色
    3. 记忆检索：基于语义相似度搜索记忆
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ensure_embedding_moe_memory()
        
    def _ensure_embedding_moe_memory(self):
        """确保智能体使用EmbeddingMoE记忆系统"""
        if not isinstance(self.agent_state.memory, EmbeddingMoEMemory):
            logger.warning("Converting agent memory to Embedding MoE system")
            
            # 保存原有记忆内容
            old_blocks = []
            if hasattr(self.agent_state.memory, 'blocks'):
                old_blocks = self.agent_state.memory.blocks
            
            # 创建新的embedding MoE记忆系统（需要异步创建）
            # 注意：这里只是标记需要转换，实际转换将在首次使用时进行
            self._needs_memory_conversion = True
            logger.info("Marked for conversion to Embedding MoE memory system")
        else:
            self._needs_memory_conversion = False
    
    async def _convert_to_embedding_moe_memory_async(self):
        """异步转换到embedding MoE记忆系统"""
        if not self._needs_memory_conversion:
            return
        
        logger.info("Converting to Embedding MoE memory system...")
        
        # 获取原有记忆内容
        persona = "AI Assistant"
        human = "User"
        
        if hasattr(self.agent_state.memory, 'blocks'):
            for block in self.agent_state.memory.blocks:
                if block.label == "persona":
                    persona = block.value or persona
                elif block.label == "human":
                    human = block.value or human
        
        # 创建新的embedding MoE记忆系统
        new_memory = await create_embedding_moe_memory_async(
            persona=persona,
            human=human,
            agent_type=self.agent_state.agent_type,
            embedding_config=self.agent_state.embedding_config,
            actor=self.user
        )
        
        # 替换记忆系统
        self.agent_state.memory = new_memory
        self._needs_memory_conversion = False
        
        logger.info("Successfully converted to Embedding MoE memory system")
    
    @trace_method
    async def step(
        self,
        input_messages: List[MessageCreate],
        max_steps: int = 10,
        run_id: Optional[str] = None,
        use_assistant_message: bool = True,
        request_start_timestamp_ns: Optional[int] = None,
        include_return_message_types: Optional[List[str]] = None,
    ) -> LettaResponse:
        """
        增强的step方法，使用embedding-based MoE记忆管理
        """
        # 确保记忆系统已转换
        if self._needs_memory_conversion:
            await self._convert_to_embedding_moe_memory_async()
        
        # 在处理消息前，基于embedding分析上下文并激活相关角色
        if input_messages and isinstance(self.agent_state.memory, EmbeddingMoEMemory):
            context = self._extract_context_from_messages(input_messages)
            await self.agent_state.memory.moe_gate_for_context_async(context, self.user)
        
        # 调用父类的step方法
        response = await super().step(
            input_messages=input_messages,
            max_steps=max_steps,
            run_id=run_id,
            use_assistant_message=use_assistant_message,
            request_start_timestamp_ns=request_start_timestamp_ns,
            include_return_message_types=include_return_message_types
        )
        
        # 处理响应后，基于embedding将重要信息添加到相关角色记忆库
        await self._post_process_memory_async(input_messages, response)
        
        return response
    
    async def _post_process_memory_async(self, input_messages: List[MessageCreate], response: LettaResponse):
        """处理响应后的记忆管理"""
        if not isinstance(self.agent_state.memory, EmbeddingMoEMemory):
            return
        
        try:
            # 提取重要信息进行记忆存储
            important_content = []
            
            # 从用户消息中提取重要信息
            for msg in input_messages:
                if hasattr(msg, 'text') and msg.text:
                    # 简单启发式：较长的消息通常包含更多重要信息
                    if len(msg.text) > 50:
                        important_content.append(msg.text)
            
            # 从响应中提取重要信息
            if hasattr(response, 'messages') and response.messages:
                for msg in response.messages:
                    if hasattr(msg, 'content') and msg.content:
                        # 提取助手的重要回复内容
                        content_text = str(msg.content)
                        if len(content_text) > 100:
                            important_content.append(content_text)
            
            # 将重要信息添加到记忆系统
            for content in important_content:
                await self.agent_state.memory.smart_add_memory_async(
                    memory_content=content,
                    context=self._extract_context_from_messages(input_messages),
                    actor=self.user,
                    tags=["conversation", "important"]
                )
                
        except Exception as e:
            logger.error(f"Error in post-processing memory: {e}")
    
    def _extract_context_from_messages(self, messages: List[MessageCreate]) -> str:
        """从消息中提取上下文"""
        context_parts = []
        
        for msg in messages:
            if hasattr(msg, 'text') and msg.text:
                context_parts.append(msg.text)
            elif hasattr(msg, 'content') and msg.content:
                context_parts.append(str(msg.content))
        
        return " ".join(context_parts)
    
    async def search_memories_by_context_async(self, query: str, max_results: int = 10) -> Dict[str, List]:
        """基于上下文搜索记忆"""
        if not isinstance(self.agent_state.memory, EmbeddingMoEMemory):
            return {}
        
        # 首先激活与查询相关的角色
        await self.agent_state.memory.moe_gate_for_context_async(query, self.user)
        
        # 搜索激活角色的记忆
        results = await self.agent_state.memory.search_role_memories_async(
            query=query,
            actor=self.user,
            top_k=max_results
        )
        
        return results
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要"""
        if not isinstance(self.agent_state.memory, EmbeddingMoEMemory):
            return {"error": "Not using Embedding MoE memory system"}
        
        return self.agent_state.memory.get_memory_statistics()


# 工厂函数
async def create_embedding_moe_agent_async(
    agent_state: AgentState,
    role_template: Optional[str] = None,
    custom_roles: Optional[List[Dict]] = None,
    interface=None,
    user=None
) -> EmbeddingMoEAgent:
    """
    创建基于embedding的MoE智能体
    
    Args:
        agent_state: 智能体状态
        role_template: 预定义角色模板名称
        custom_roles: 自定义角色列表
        interface: 交互接口
        user: 用户对象
    """
    
    # 确保有embedding配置
    if not agent_state.embedding_config:
        from letta.schemas.embedding_config import EmbeddingConfig
        agent_state.embedding_config = EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
            embedding_chunk_size=300
        )
    
    # 获取persona和human信息
    persona = "我是一个基于embedding的MoE智能助手，能够智能管理不同领域的专业知识。"
    human = "用户正在测试基于embedding的MoE记忆系统。"
    
    if hasattr(agent_state, 'memory') and hasattr(agent_state.memory, 'blocks'):
        for block in agent_state.memory.blocks:
            if block.label == "persona":
                persona = block.value or persona
            elif block.label == "human":
                human = block.value or human
    
    # 创建embedding MoE记忆系统
    embedding_memory = await create_embedding_moe_memory_async(
        persona=persona,
        human=human,
        agent_type=agent_state.agent_type,
        embedding_config=agent_state.embedding_config,
        actor=user
    )
    
    # 如果指定了角色模板，添加模板角色
    if role_template:
        template_config = RoleTemplates.get_role_template(role_template)
        if template_config and user:
            # 将模板角色转换为embedding角色配置
            from letta.schemas.embedding_moe_memory import EmbeddingRoleConfig, EmbeddingRoleMemoryRepository
            
            embedding_role_config = EmbeddingRoleConfig(
                role_id=template_config.role_id,
                role_name=template_config.role_name,
                role_description=template_config.description,
                keywords=template_config.keywords,
                max_memory_size=template_config.max_memory_size,
                activation_threshold=0.3
            )
            
            repository = EmbeddingRoleMemoryRepository(config=embedding_role_config)
            embedding_memory.role_repositories[template_config.role_id] = repository
            
            # 生成角色embedding
            await embedding_memory._ensure_role_embeddings_async(user)
    
    # 添加自定义角色
    if custom_roles and user:
        for role_data in custom_roles:
            from letta.schemas.embedding_moe_memory import EmbeddingRoleConfig, EmbeddingRoleMemoryRepository
            
            embedding_role_config = EmbeddingRoleConfig(**role_data)
            repository = EmbeddingRoleMemoryRepository(config=embedding_role_config)
            embedding_memory.role_repositories[role_data["role_id"]] = repository
        
        # 生成所有角色的embedding
        await embedding_memory._ensure_role_embeddings_async(user)
    
    # 更新agent_state的记忆系统
    agent_state.memory = embedding_memory
    
    # 创建智能体
    agent = EmbeddingMoEAgent(
        agent_state=agent_state,
        interface=interface,
        user=user,
        # 其他必要参数...
    )
    
    return agent
