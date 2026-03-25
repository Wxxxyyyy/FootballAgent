# -*- coding: utf-8 -*-
"""
认证路由：注册、登录（JWT）、当前用户。
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.user_service import UserService, get_db

router = APIRouter(tags=["认证"])

SECRET_KEY = os.getenv("JWT_SECRET", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_MINUTES = int(os.getenv("JWT_ACCESS_MINUTES", "60"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class RegisterBody(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    email: str | None = None


class LoginBody(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    username: str
    email: str | None = None


def _encode_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_MINUTES)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = int(payload.get("sub", "0"))
    except (JWTError, ValueError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "无效凭证")
    svc = UserService()
    user = await svc.get_user_by_id(session, uid)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户不存在")
    return user


@router.post("/auth/register", response_model=UserOut)
async def register(body: RegisterBody, session: Annotated[AsyncSession, Depends(get_db)]):
    """新用户注册。"""
    svc = UserService()
    try:
        u = await svc.create_user(session, body.username, body.password, body.email)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return UserOut(id=u["id"], username=u["username"], email=u.get("email"))


@router.post("/auth/login", response_model=TokenOut)
async def login(body: LoginBody, session: Annotated[AsyncSession, Depends(get_db)]):
    """校验密码并签发 JWT（Swagger 中请用 Authorize 填入 Bearer token）。"""
    svc = UserService()
    user = await svc.authenticate(session, body.username, body.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")
    token = _encode_token(user["id"])
    return TokenOut(access_token=token)


@router.get("/auth/me", response_model=UserOut)
async def me(current: Annotated[dict, Depends(get_current_user)]):
    """从 JWT 解析当前用户。"""
    return UserOut(
        id=current["id"], username=current["username"], email=current.get("email")
    )
