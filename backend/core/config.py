# -*- coding: utf-8 -*-
"""
全局配置（pydantic-settings）
从环境变量 / .env 加载 MySQL、Neo4j、Redis、LLM、JWT 与应用参数。
"""
from functools import lru_cache
from typing import Optional
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用级配置，字段名与 .env 中的 KEY 对应（不区分大小写）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- MySQL ----
    mysql_host: str = Field(default="127.0.0.1", description="MySQL 主机")
    mysql_port: int = Field(default=3306, description="MySQL 端口")
    mysql_user: str = Field(default="root", description="MySQL 用户名")
    mysql_password: str = Field(default="", description="MySQL 密码")
    mysql_database: str = Field(default="football", description="MySQL 库名")

    # ---- Neo4j ----
    neo4j_url: str = Field(default="bolt://127.0.0.1:7687", description="Neo4j Bolt URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j 用户名")
    neo4j_password: str = Field(default="", description="Neo4j 密码")

    # ---- Redis ----
    redis_host: str = Field(default="127.0.0.1", description="Redis 主机")
    redis_port: int = Field(default=6379, description="Redis 端口")
    redis_password: Optional[str] = Field(default=None, description="Redis 密码，无则留空")
    redis_db: int = Field(default=0, ge=0, description="Redis 逻辑库编号")

    # ---- LLM ----
    llm_api_key: str = Field(default="", description="大模型 API Key")
    llm_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI 兼容 Base URL")
    llm_model: str = Field(default="gpt-4o-mini", description="默认模型名")

    # ---- JWT ----
    jwt_secret: str = Field(default="change-me-in-production", min_length=8, description="JWT 签名密钥")
    jwt_algorithm: str = Field(default="HS256", description="JWT 算法")
    jwt_expire_minutes: int = Field(default=60 * 24, ge=1, description="Access Token 过期（分钟）")

    # ---- App ----
    app_debug: bool = Field(default=False, description="调试模式")
    app_host: str = Field(default="0.0.0.0", description="监听地址")
    app_port: int = Field(default=8000, ge=1, le=65535, description="监听端口")

    # OAuth2 文档用（Swagger「Authorize」）；可与实际登录路由一致
    oauth2_token_url: str = Field(default="/api/v1/auth/login", description="OAuth2 token 文档路径")

    @field_validator("neo4j_url")
    @classmethod
    def _strip_neo4j_url(cls, v: str) -> str:
        return v.strip()

    @property
    def sqlalchemy_database_uri(self) -> str:
        """SQLAlchemy 异步连接串（aiomysql）；用户名密码经 URL 编码以兼容特殊字符。"""
        user = quote_plus(self.mysql_user)
        pwd = quote_plus(self.mysql_password)
        host = self.mysql_host
        port = self.mysql_port
        db = self.mysql_database
        return f"mysql+aiomysql://{user}:{pwd}@{host}:{port}/{db}"

    @property
    def redis_url(self) -> str:
        """redis.asyncio 可用的 URL。"""
        auth = ""
        if self.redis_password:
            from urllib.parse import quote

            auth = f":{quote(self.redis_password, safe='')}@"
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """返回进程内单例 Settings，避免重复解析 .env。"""
    return Settings()
