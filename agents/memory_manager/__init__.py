# -*- coding: utf-8 -*-
"""
记忆管理模块
- Compaction: 对话超长时压缩旧消息为摘要，Embedding 存入 ChromaDB
- Retrieval:  按需从 ChromaDB 检索历史摘要注入上下文
"""
