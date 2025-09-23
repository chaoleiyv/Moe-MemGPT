"""
Role Templates for Embedding-based MoE Memory System

This module provides pre-defined role templates and utilities for creating
commonly used role configurations in different scenarios.
All templates now use embedding-based semantic matching.
"""

from typing import List, Dict, Any, Optional
from letta.schemas.embedding_moe_memory import EmbeddingRoleConfig, RoleType


class RoleTemplates:
    """预定义的角色模板集合（基于Embedding）"""
    
    @staticmethod
    def get_personal_assistant_roles() -> List[EmbeddingRoleConfig]:
        """个人助理场景的角色配置"""
        return [
            EmbeddingRoleConfig(
                role_id="general_assistant",
                role_type=RoleType.PERSONAL,
                role_name="通用助手",
                role_description="处理日常对话和一般性问题，包括基本的问候、帮助请求和常规聊天交流",
                keywords=["hello", "help", "question", "chat", "你好", "帮助"],
                activation_threshold=0.3
            ),
            EmbeddingRoleConfig(
                role_id="schedule_manager",
                role_type=RoleType.PROFESSIONAL,
                role_name="日程管理",
                role_description="管理日程安排、会议预约、时间提醒和日历相关的所有事务",
                keywords=["schedule", "meeting", "appointment", "calendar", "reminder", "日程", "会议", "提醒"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="personal_life",
                role_type=RoleType.PERSONAL,
                role_name="生活助手",
                role_description="处理个人生活、兴趣爱好、家庭关系和日常生活相关的各种事务",
                keywords=["hobby", "family", "friend", "personal", "life", "爱好", "家庭", "生活"],
                activation_threshold=0.3
            ),
            EmbeddingRoleConfig(
                role_id="health_wellness",
                role_type=RoleType.HEALTH,
                role_name="健康顾问",
                role_description="提供健康管理、运动指导、饮食建议和整体健康生活方式的建议",
                keywords=["health", "exercise", "diet", "wellness", "fitness", "健康", "运动", "饮食"],
                activation_threshold=0.4
            )
        ]
    
    @staticmethod
    def get_professional_assistant_roles() -> List[EmbeddingRoleConfig]:
        """专业工作助手场景的角色配置"""
        return [
            EmbeddingRoleConfig(
                role_id="project_manager",
                role_type=RoleType.PROFESSIONAL,
                role_name="项目经理",
                role_description="负责项目管理、进度跟踪、任务分配、团队协调和项目交付的全流程管理",
                keywords=["project", "task", "deadline", "milestone", "team", "progress", "项目", "任务", "截止日期"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="technical_lead",
                role_type=RoleType.TECHNICAL,
                role_name="技术负责人",
                role_description="处理技术决策、系统架构设计、代码审查、技术选型和开发团队的技术指导",
                keywords=["architecture", "design", "code", "review", "technical", "development", "架构", "设计", "代码"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="business_analyst",
                role_type=RoleType.PROFESSIONAL,
                role_name="业务分析师",
                role_description="分析业务需求、市场趋势、竞争对手、商业策略和业务流程优化",
                keywords=["business", "analysis", "requirement", "strategy", "market", "商业", "分析", "需求"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="communication_hub",
                role_type=RoleType.SOCIAL,
                role_name="沟通协调",
                role_description="处理团队内外沟通、客户关系维护、会议组织和跨部门协调工作",
                keywords=["communication", "meeting", "client", "discussion", "email", "沟通", "客户", "讨论"],
                activation_threshold=0.3
            )
        ]
    
    @staticmethod
    def get_developer_assistant_roles() -> List[EmbeddingRoleConfig]:
        """开发者助手场景的角色配置"""
        return [
            EmbeddingRoleConfig(
                role_id="coding_expert",
                role_type=RoleType.TECHNICAL,
                role_name="编程专家",
                role_description="专注于代码编写、算法设计、程序调试、性能优化和编程最佳实践",
                keywords=["code", "programming", "debug", "optimize", "algorithm", "编程", "代码", "调试"],
                activation_threshold=0.5
            ),
            EmbeddingRoleConfig(
                role_id="devops_engineer",
                role_type=RoleType.TECHNICAL,
                role_name="运维工程师",
                role_description="负责系统部署、监控运维、基础设施管理、CI/CD流程和云服务配置",
                keywords=["deploy", "monitor", "infrastructure", "server", "docker", "kubernetes", "部署", "监控"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="qa_tester",
                role_type=RoleType.TECHNICAL,
                role_name="质量保证",
                role_description="软件测试、质量控制、缺陷跟踪、自动化测试和质量流程管理",
                keywords=["test", "testing", "bug", "quality", "qa", "automation", "测试", "质量", "错误"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="documentation",
                role_type=RoleType.CREATIVE,
                role_name="文档管理",
                role_description="技术文档编写、API文档维护、用户手册创建和知识库管理",
                keywords=["documentation", "doc", "readme", "guide", "manual", "文档", "说明", "指南"],
                activation_threshold=0.3
            )
        ]
    
    @staticmethod
    def get_student_tutor_roles() -> List[EmbeddingRoleConfig]:
        """学生导师场景的角色配置"""
        return [
            EmbeddingRoleConfig(
                role_id="academic_advisor",
                role_type=RoleType.ACADEMIC,
                role_name="学术顾问",
                role_description="提供学术指导、研究方向建议、课程规划和学术发展路径咨询",
                keywords=["academic", "research", "study", "course", "thesis", "学术", "研究", "学习"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="skill_mentor",
                role_type=RoleType.TECHNICAL,
                role_name="技能导师",
                role_description="专业技能培养、实践项目指导、技能评估和学习方法优化",
                keywords=["skill", "practice", "training", "tutorial", "learning", "技能", "练习", "培训"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="career_counselor",
                role_type=RoleType.PROFESSIONAL,
                role_name="职业规划",
                role_description="职业发展规划、就业指导、面试准备、简历优化和职场技能培养",
                keywords=["career", "job", "interview", "resume", "professional", "职业", "工作", "面试"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="life_coach",
                role_type=RoleType.PERSONAL,
                role_name="生活导师",
                role_description="生活平衡指导、时间管理、个人成长、目标设定和心理健康支持",
                keywords=["life", "balance", "time", "management", "personal", "growth", "生活", "平衡", "成长"],
                activation_threshold=0.3
            )
        ]
    
    @staticmethod
    def get_creative_assistant_roles() -> List[EmbeddingRoleConfig]:
        """创意助手场景的角色配置"""
        return [
            EmbeddingRoleConfig(
                role_id="creative_director",
                role_type=RoleType.CREATIVE,
                role_name="创意总监",
                role_description="创意策划、设计指导、艺术方向制定和创意项目的整体把控",
                keywords=["creative", "design", "art", "visual", "concept", "创意", "设计", "艺术"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="content_writer",
                role_type=RoleType.CREATIVE,
                role_name="内容创作",
                role_description="文案写作、内容策划、故事创作、编辑校对和内容营销策略",
                keywords=["writing", "content", "copy", "editor", "story", "写作", "内容", "文案"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="media_producer",
                role_type=RoleType.CREATIVE,
                role_name="媒体制作",
                role_description="视频制作、音频编辑、多媒体内容创作和媒体项目管理",
                keywords=["video", "audio", "media", "production", "editing", "视频", "音频", "制作"],
                activation_threshold=0.4
            ),
            EmbeddingRoleConfig(
                role_id="brand_strategist",
                role_type=RoleType.PROFESSIONAL,
                role_name="品牌策略",
                role_description="品牌定位、市场推广策略、用户体验设计和品牌形象管理",
                keywords=["brand", "marketing", "strategy", "user", "experience", "品牌", "营销", "策略"],
                activation_threshold=0.4
            )
        ]


class RoleConfigBuilder:
    """角色配置构建器，用于创建自定义角色配置"""
    
    def __init__(self):
        self.configs = []
    
    def add_role(
        self,
        role_id: str,
        role_name: str,
        role_type: RoleType,
        description: str,
        keywords: List[str],
        activation_threshold: float = 0.3,
        max_memory_size: Optional[int] = 50,
        retention_strategy: str = "semantic"
    ) -> "RoleConfigBuilder":
        """添加角色配置"""
        config = EmbeddingRoleConfig(
            role_id=role_id,
            role_type=role_type,
            role_name=role_name,
            role_description=description,
            keywords=keywords,
            activation_threshold=activation_threshold,
            max_memory_size=max_memory_size,
            memory_retention_strategy=retention_strategy
        )
        self.configs.append(config)
        return self
    
    def build(self) -> List[EmbeddingRoleConfig]:
        """构建角色配置列表"""
        return self.configs.copy()


def get_role_template(template_name: str) -> EmbeddingRoleConfig:
    """
    根据模板名称获取单个角色配置（用于简单场景）
    
    Args:
        template_name: 模板名称
            - "personal_assistant": 个人助理
            - "professional_assistant": 专业工作助手  
            - "technical_expert": 技术专家
            - "creative_assistant": 创意助手
            
    Returns:
        EmbeddingRoleConfig: 单个角色配置
    """
    simple_templates = {
        "personal_assistant": EmbeddingRoleConfig(
            role_id="personal_assistant",
            role_type=RoleType.PERSONAL,
            role_name="个人助手",
            role_description="贴心的个人生活助手，帮助处理日常生活事务、健康管理、娱乐休闲、人际关系等个人生活相关的需求。语言亲切自然，关注用户的个人感受和生活品质。",
            keywords=["生活", "个人", "健康", "娱乐", "休闲", "家庭", "朋友", "personal", "life", "family"],
            activation_threshold=0.3
        ),
        "professional_assistant": EmbeddingRoleConfig(
            role_id="professional_assistant",
            role_type=RoleType.PROFESSIONAL,
            role_name="专业助手",
            role_description="专业工作助手，擅长处理商务沟通、项目管理、技术咨询、工作流程优化等专业领域的任务。具有严谨的工作态度和专业的沟通方式，能够提供高效的工作解决方案。",
            keywords=["工作", "项目", "业务", "专业", "技术", "管理", "咨询", "work", "project", "business"],
            activation_threshold=0.3
        ),
        "technical_expert": EmbeddingRoleConfig(
            role_id="technical_expert",
            role_type=RoleType.TECHNICAL,
            role_name="技术专家",
            role_description="资深技术专家，精通编程开发、系统架构、技术调研、问题排查等技术领域。能提供深入的技术分析和解决方案，熟悉各种开发工具和最佳实践。",
            keywords=["编程", "开发", "技术", "代码", "架构", "算法", "系统", "programming", "development", "technical"],
            activation_threshold=0.4
        ),
        "creative_assistant": EmbeddingRoleConfig(
            role_id="creative_assistant",
            role_type=RoleType.CREATIVE,
            role_name="创意助手",
            role_description="富有创造力的助手，专注于创意策划、内容创作、设计思维和艺术表达。能够激发灵感，提供创新的解决方案和独特的视角。",
            keywords=["创意", "设计", "艺术", "创作", "灵感", "创新", "creative", "design", "art", "innovation"],
            activation_threshold=0.4
        )
    }
    
    if template_name not in simple_templates:
        available = list(simple_templates.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available templates: {available}")
    
    return simple_templates[template_name]


def get_role_template_set(template_name: str) -> List[EmbeddingRoleConfig]:
    """
    根据模板名称获取角色配置集合（用于复杂场景）
    
    Args:
        template_name: 模板名称
            - "personal_assistant_set": 个人助理集合
            - "professional_assistant_set": 专业工作助手集合
            - "developer_assistant_set": 开发者助手集合
            - "student_tutor_set": 学生导师集合
            - "creative_assistant_set": 创意助手集合
            
    Returns:
        List[EmbeddingRoleConfig]: 角色配置列表
    """
    template_sets = {
        "personal_assistant_set": RoleTemplates.get_personal_assistant_roles,
        "professional_assistant_set": RoleTemplates.get_professional_assistant_roles,
        "developer_assistant_set": RoleTemplates.get_developer_assistant_roles,
        "student_tutor_set": RoleTemplates.get_student_tutor_roles,
        "creative_assistant_set": RoleTemplates.get_creative_assistant_roles,
    }
    
    if template_name not in template_sets:
        available = list(template_sets.keys())
        raise ValueError(f"Unknown template set '{template_name}'. Available template sets: {available}")
    
    return template_sets[template_name]()


def create_custom_role_set(scenario_description: str, role_specs: List[Dict[str, Any]]) -> List[EmbeddingRoleConfig]:
    """
    根据场景描述和角色规格创建自定义角色集合
    
    Args:
        scenario_description: 场景描述
        role_specs: 角色规格列表，每个包含 name, type, description, keywords 等
        
    Returns:
        List[EmbeddingRoleConfig]: 自定义角色配置列表
    """
    builder = RoleConfigBuilder()
    
    for i, spec in enumerate(role_specs):
        role_id = f"custom_{i}_{spec['name'].lower().replace(' ', '_')}"
        builder.add_role(
            role_id=role_id,
            role_name=spec['name'],
            role_type=RoleType(spec.get('type', 'custom')),
            description=spec.get('description', ''),
            keywords=spec.get('keywords', []),
            activation_threshold=spec.get('activation_threshold', 0.3),
            max_memory_size=spec.get('max_memory_size', 50),
            retention_strategy=spec.get('retention_strategy', 'semantic')
        )
    
    return builder.build()