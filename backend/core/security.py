# -*- coding: utf-8 -*-
"""
JWT 与密码哈希
- Access Token 签发与校验
- bcrypt 密码存储
- FastAPI Depends 获取当前用户主体
"""
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from .config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=get_settings().oauth2_token_url,
    auto_error=True,
)


class TokenPayload(BaseModel):
    """JWT 载荷中业务常用字段（可按业务扩展）。"""

    sub: str = Field(..., description="用户主键或唯一标识")
    username: Optional[str] = Field(default=None, description="展示名")
    exp: Optional[int] = Field(default=None, description="过期时间（unix 秒）")


def hash_password(plain_password: str) -> str:
    """对明文密码做单向哈希，入库前调用。"""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    """校验明文与存储哈希是否匹配。"""
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(
    subject: str,
    *,
    extra_claims: Optional[dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """签发 JWT access token（HS256）。"""
    settings = get_settings()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.jwt_expire_minutes)
    )
    payload: dict[str, Any] = {"sub": subject, "exp": expire}
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> dict[str, Any]:
    """
    解码并校验 JWT。
    签名无效或过期时抛出 jose.JWTError，由上层转换为 HTTP 响应。
    """
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> TokenPayload:
    """
    FastAPI 依赖：从 Authorization Bearer 解析当前用户。
    后续可在路由中查询数据库补全 User ORM。
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = verify_token(token)
        sub = payload.get("sub")
        if sub is None or not isinstance(sub, str):
            raise credentials_exception
        return TokenPayload(
            sub=sub,
            username=payload.get("username") if isinstance(payload.get("username"), str) else None,
            exp=payload.get("exp") if isinstance(payload.get("exp"), int) else None,
        )
    except JWTError:
        raise credentials_exception from None
