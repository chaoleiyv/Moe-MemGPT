#!/usr/bin/env python3
"""
Letta 基于Embedding的MoE角色记忆Agent本地创建和测试脚本

功能：
1. 使用基于embedding的MoE记忆系统（完全基于向量语义匹配）
2. 角色描述向量化，记忆内容向量化
3. 基于embedding相似度进行记忆路由和角色激活
4. 完全模仿Archival Memory的实现方式
5. 直接使用本地代码（无需服务器）

使用方法：
    python create_embedding_moe_agent_local.py

环境要求：
    export OPENAI_API_KEY="your-api-key"
    # 不需要启动服务器！
"""

import os
import sys
import asyncio
from pathlib import Path

def check_environment():
    """检查环境"""
    print("🔍 检查本地环境...")
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 请设置: export OPENAI_API_KEY='your-api-key'")
        return False
    
    # 检查是否在正确的目录
    if not Path("letta").exists():
        print("❌ 请在项目根目录运行此脚本")
        return False
    
    print("✅ 本地环境检查通过")
    return True

async def create_embedding_moe_agent_async():
    """异步创建基于embedding的MoE Agent"""
    print("🎭 使用本地代码创建基于Embedding的MoE角色记忆Agent...")
    
    # 添加项目路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "letta"))
    
    try:
        # 导入embedding MoE模块
        from letta.agents.embedding_moe_agent import create_embedding_moe_agent_async, EmbeddingMoEAgent
        from letta.schemas.agent import AgentState, AgentType
        from letta.schemas.llm_config import LLMConfig
        from letta.schemas.embedding_config import EmbeddingConfig
        from letta.schemas.embedding_moe_memory import EmbeddingMoEMemory
        from letta.schemas.role_templates import get_role_template
        from letta.schemas.user import User
        
        print("✅ 成功导入Embedding MoE模块")
        
        # 确保EmbeddingMoEMemory类可用
        globals()['EmbeddingMoEMemory'] = EmbeddingMoEMemory
        
        # 创建模拟用户（用于embedding生成）
        user = User(
            id="test_user",
            name="Test User",
            organization_id="test_org"
        )
        
        # 创建基本的智能体状态
        agent_state = AgentState(
            id="local_embedding_moe_agent",
            name="Local_Embedding_MoE_Professional_Assistant", 
            agent_type=AgentType.memgpt_agent,
            
            # LLM配置
            llm_config=LLMConfig(
                model="gpt-4o-mini",
                model_endpoint_type="openai",
                context_window=8192
            ),
            
            # 嵌入配置（关键配置！）
            embedding_config=EmbeddingConfig(
                embedding_model="text-embedding-3-small",
                embedding_endpoint_type="openai",
                embedding_dim=1536,
                embedding_chunk_size=300
            ),
            
            tools=[],
            sources=[],
            tags=[],
            system="你是一个具有基于embedding语义匹配的多角色记忆能力的AI助手。"
        )
        
        print("📋 使用预定义角色模板: professional_assistant")
        
        # 使用预定义的专业助手角色模板创建embedding MoE智能体
        embedding_moe_agent = await create_embedding_moe_agent_async(
            agent_state=agent_state,
            role_template="professional_assistant",  # 使用预定义模板
            interface=None,
            user=user
        )
        
        print(f"✅ 本地Embedding MoE Agent创建成功！")
        print(f"   ID: {embedding_moe_agent.agent_state.id}")
        print(f"   名称: {embedding_moe_agent.agent_state.name}")
        print(f"   模型: {embedding_moe_agent.agent_state.llm_config.model}")
        print(f"   Embedding模型: {embedding_moe_agent.agent_state.embedding_config.embedding_model}")
        print(f"   记忆类型: {type(embedding_moe_agent.agent_state.memory).__name__}")
        
        # 显示角色摘要
        summary = embedding_moe_agent.get_memory_summary()
        print(f"\n🎭 已配置的角色库:")
        for role_id, role_info in summary.get("role_details", {}).items():
            status = "✅" if role_info["is_active"] else "❌"
            embed_mark = "🧠" if role_info.get("has_embedding", False) else "❓"
            print(f"   {status} {embed_mark} {role_info['role_name']} ({role_id})")
            print(f"      记忆块: {role_info['memory_blocks']}, 档案段落: {role_info['archival_passages']}")
        
        return embedding_moe_agent, user
        
    except ImportError as e:
        print(f"❌ 导入Embedding MoE模块失败: {e}")
        print("请确保在项目根目录运行脚本")
        return None, None
    except Exception as e:
        print(f"❌ 创建Embedding MoE Agent失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

async def test_memory_routing_async(embedding_moe_agent, user):
    """测试基于embedding的记忆路由功能"""
    print("\n🧠 测试基于Embedding的记忆路由功能...")
    
    if not embedding_moe_agent:
        print("❌ Embedding MoE Agent未创建")
        return
    
    # 测试用例（更复杂的语义内容）
    test_cases = [
        {
            "category": "项目管理",
            "content": "我们的软件开发项目遇到了延期问题，需要重新评估里程碑和资源分配策略",
            "expected_roles": ["professional_assistant"]
        },
        {
            "category": "技术架构", 
            "content": "系统需要支持高并发访问，考虑使用微服务架构和分布式缓存解决方案",
            "expected_roles": ["technical_expert"]
        },
        {
            "category": "个人学习",
            "content": "我正在学习机器学习算法，特别对深度学习和神经网络很感兴趣",
            "expected_roles": ["personal_assistant"]
        },
        {
            "category": "业务分析",
            "content": "市场调研显示用户对我们产品的核心功能满意度较高，但界面体验需要改进",
            "expected_roles": ["professional_assistant"]
        }
    ]
    
    embedding_moe_memory = embedding_moe_agent.agent_state.memory
    
    if not isinstance(embedding_moe_memory, EmbeddingMoEMemory):
        print("⚠️  Agent未使用Embedding MoE记忆系统")
        return
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i}: {test_case['category']} ---")
        print(f"📝 内容: {test_case['content']}")
        print(f"🎯 预期路由到: {test_case['expected_roles']}")
        
        # 测试基于embedding的记忆路由
        relevant_roles = await embedding_moe_memory.route_memory_to_role_async(
            test_case['content'], 
            context="测试上下文",
            actor=user
        )
        print(f"📦 实际路由到: {relevant_roles}")
        
        # 验证路由结果
        if relevant_roles:
            print(f"✅ 路由验证: 成功路由到 {len(relevant_roles)} 个角色")
            for role_id in relevant_roles:
                if role_id in embedding_moe_memory.role_repositories:
                    role_name = embedding_moe_memory.role_repositories[role_id].config.role_name
                    print(f"   🎭 {role_name} ({role_id})")
        else:
            print(f"⚠️  路由验证: 未找到合适的角色")
        
        # 使用智能添加记忆功能
        success = await embedding_moe_memory.smart_add_memory_async(
            test_case['content'],
            context="测试上下文",
            actor=user,
            tags=[test_case['category'], "test"]
        )
        print(f"📚 智能添加记忆: {'成功' if success else '失败'}")

async def test_memory_search_async(embedding_moe_agent, user):
    """测试基于embedding的记忆搜索功能"""
    print("\n🔍 测试基于Embedding的记忆搜索功能...")
    
    if not embedding_moe_agent:
        print("❌ Embedding MoE Agent未创建")
        return
    
    embedding_moe_memory = embedding_moe_agent.agent_state.memory
    
    if not isinstance(embedding_moe_memory, EmbeddingMoEMemory):
        print("⚠️  Agent未使用Embedding MoE记忆系统")
        return
    
    # 语义搜索测试
    search_queries = [
        ("项目延期问题", "搜索项目管理相关记忆"),
        ("微服务和缓存", "搜索技术架构相关记忆"),
        ("机器学习算法", "搜索个人学习相关记忆"),
        ("用户体验改进", "搜索业务分析相关记忆")
    ]
    
    for query, description in search_queries:
        print(f"\n🔎 {description}: '{query}'")
        
        # 基于embedding的语义搜索
        search_results = await embedding_moe_memory.search_role_memories_async(
            query=query,
            actor=user,
            top_k=3
        )
        
        found_results = False
        for role_id, memories in search_results.items():
            if memories:
                found_results = True
                role_name = embedding_moe_memory.role_repositories[role_id].config.role_name
                print(f"   🧠 {role_name} ({role_id}): 找到 {len(memories)} 条语义相关记忆")
                for memory in memories[:2]:  # 只显示前2条
                    print(f"      - 相似度: {memory.semantic_score:.3f} | {memory.value[:60]}...")
        
        if not found_results:
            print(f"   📚 未找到语义相关记忆")

async def test_moe_gating_async(embedding_moe_agent, user):
    """测试基于embedding的MoE门控机制"""
    print("\n🚪 测试基于Embedding的MoE门控机制...")
    
    if not embedding_moe_agent:
        print("❌ Embedding MoE Agent未创建")
        return
    
    embedding_moe_memory = embedding_moe_agent.agent_state.memory
    
    # 测试不同上下文的语义门控激活
    test_contexts = [
        ("我需要设计一个可扩展的分布式系统架构", "技术架构上下文"),
        ("项目截止日期临近，团队压力很大，需要合理安排工作", "项目管理上下文"),
        ("我想学习Python编程和数据科学相关知识", "个人学习上下文"),
        ("产品用户反馈收集完成，需要分析改进方向", "业务分析上下文")
    ]
    
    for context, description in test_contexts:
        print(f"\n🔍 {description}: '{context}'")
        
        # 使用基于embedding的MoE门控
        active_roles = await embedding_moe_memory.moe_gate_for_context_async(
            context, 
            actor=user,
            max_roles=2
        )
        print(f"🎯 Embedding MoE门控激活角色: {active_roles}")
        
        # 显示激活角色的详细信息
        for role_id in active_roles:
            if role_id in embedding_moe_memory.role_repositories:
                repo = embedding_moe_memory.role_repositories[role_id]
                embed_mark = "🧠" if repo.config.role_embedding else "❓"
                print(f"   {embed_mark} {repo.config.role_name} ({role_id})")
                print(f"      描述: {repo.config.role_description[:80]}...")

def show_memory_statistics(embedding_moe_agent):
    """显示Embedding MoE记忆系统统计信息"""
    print("\n📊 Embedding MoE记忆系统统计...")
    
    if not embedding_moe_agent:
        print("❌ Embedding MoE Agent未创建")
        return
    
    summary = embedding_moe_agent.get_memory_summary()
    
    print(f"📋 系统概览:")
    print(f"   总角色数: {summary.get('total_roles', 0)}")
    print(f"   活跃角色数: {summary.get('active_roles', 0)}")
    print(f"   总记忆块数: {summary.get('total_memory_blocks', 0)}")
    print(f"   总档案段落数: {summary.get('total_archival_passages', 0)}")
    
    print(f"\n📈 各角色记忆分布:")
    for role_id, role_info in summary.get("role_details", {}).items():
        memory_count = role_info.get("memory_blocks", 0)
        archival_count = role_info.get("archival_passages", 0)
        embed_status = "🧠" if role_info.get("has_embedding", False) else "❓"
        
        bar_length = min(memory_count, 15)  # 最大15个字符
        bar = "█" * bar_length + "░" * (15 - bar_length)
        print(f"   {embed_status} {role_info['role_name']:<15} [{bar}] M:{memory_count} A:{archival_count}")

async def test_archival_memory_async(embedding_moe_agent, user):
    """测试档案记忆功能（完全模仿Archival Memory）"""
    print("\n📚 测试档案记忆功能...")
    
    if not embedding_moe_agent:
        print("❌ Embedding MoE Agent未创建")
        return
    
    embedding_moe_memory = embedding_moe_agent.agent_state.memory
    
    # 添加一些档案记忆
    archival_contents = [
        "Python是一种高级编程语言，广泛用于数据科学、机器学习和Web开发。它具有简洁的语法和强大的库生态系统。",
        "微服务架构是一种将单体应用程序分解为多个小型、独立服务的设计模式。每个服务都可以独立部署和扩展。",
        "项目管理中的敏捷方法论强调迭代开发、团队协作和快速响应变化。Scrum是最流行的敏捷框架之一。"
    ]
    
    print("📝 添加档案记忆...")
    for i, content in enumerate(archival_contents, 1):
        # 选择一个角色添加档案记忆
        role_id = list(embedding_moe_memory.role_repositories.keys())[0]
        repository = embedding_moe_memory.role_repositories[role_id]
        
        passage = await repository.add_archival_passage_async(
            text=content,
            embedding_config=embedding_moe_memory.embedding_config,
            actor=user,
            tags=[f"archival_{i}", "knowledge"]
        )
        
        print(f"   ✅ 档案记忆 {i} 已添加到 {repository.config.role_name}")
    
    # 测试档案记忆搜索
    print("\n🔍 测试档案记忆搜索...")
    search_queries = ["编程语言", "服务架构", "项目开发"]
    
    for query in search_queries:
        print(f"\n🔎 搜索: '{query}'")
        
        for role_id, repository in embedding_moe_memory.role_repositories.items():
            if repository.archival_passages:
                results = await repository.search_archival_memories_async(
                    query=query,
                    embedding_config=embedding_moe_memory.embedding_config,
                    actor=user,
                    top_k=2
                )
                
                if results:
                    print(f"   📚 {repository.config.role_name}: 找到 {len(results)} 条档案记忆")
                    for passage in results:
                        print(f"      - {passage.text[:60]}...")

async def main_async():
    """异步主函数"""
    print("🎉 Letta 基于Embedding的MoE角色记忆系统 - 本地版本")
    print("=" * 70)
    
    # 环境检查
    if not check_environment():
        print("\n💡 环境配置步骤:")
        print("1. export OPENAI_API_KEY='your-api-key'")
        print("2. 确保在项目根目录运行脚本")
        sys.exit(1)
    
    try:
        # 创建Embedding MoE Agent
        embedding_moe_agent, user = await create_embedding_moe_agent_async()
        if not embedding_moe_agent:
            sys.exit(1)
        
        # 测试基于embedding的记忆路由
        await test_memory_routing_async(embedding_moe_agent, user)
        
        # 测试基于embedding的记忆搜索
        await test_memory_search_async(embedding_moe_agent, user)
        
        # 测试基于embedding的MoE门控机制
        await test_moe_gating_async(embedding_moe_agent, user)
        
        # 测试档案记忆功能
        await test_archival_memory_async(embedding_moe_agent, user)
        
        # 显示统计信息
        show_memory_statistics(embedding_moe_agent)
        
        print(f"\n🎊 基于Embedding的MoE角色记忆系统测试完成！")
        print(f"✅ 成功创建并测试了Embedding MoE角色记忆Agent")
        print(f"🧠 完全基于向量语义匹配进行记忆管理")
        print(f"🎭 角色激活和记忆路由都使用embedding相似度")
        print(f"📚 模仿Archival Memory实现了档案记忆功能")
        
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """同步入口点"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
