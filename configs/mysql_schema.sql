-- ⚽ Football Agent - MySQL 表结构定义
-- 执行方式: mysql -u root -p football_agent < configs/mysql_schema.sql

-- 创建数据库（如不存在）
CREATE DATABASE IF NOT EXISTS football_agent
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE football_agent;

-- ============================================
-- 1. 比赛记录表
-- ============================================
CREATE TABLE IF NOT EXISTS matches (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    div             VARCHAR(10)    COMMENT '联赛代码 (E0=英超, F1=法甲, D1=德甲, I1=意甲, SP1=西甲)',
    season          VARCHAR(20)    COMMENT '赛季 (如 2024-2025)',
    match_date      DATE           COMMENT '比赛日期',
    match_time      VARCHAR(10)    COMMENT '比赛时间',
    home_team       VARCHAR(100)   COMMENT '主队名称',
    away_team       VARCHAR(100)   COMMENT '客队名称',
    fthg            SMALLINT       COMMENT '全场主队进球',
    ftag            SMALLINT       COMMENT '全场客队进球',
    ftr             CHAR(1)        COMMENT '全场结果 (H=主胜, D=平, A=客胜)',
    hthg            SMALLINT       COMMENT '半场主队进球',
    htag            SMALLINT       COMMENT '半场客队进球',
    htr             CHAR(1)        COMMENT '半场结果',
    referee         VARCHAR(100)   COMMENT '裁判',
    hs              SMALLINT       COMMENT '主队射门数',
    as_shots        SMALLINT       COMMENT '客队射门数',
    hst             SMALLINT       COMMENT '主队射正数',
    ast             SMALLINT       COMMENT '客队射正数',
    hf              SMALLINT       COMMENT '主队犯规数',
    af              SMALLINT       COMMENT '客队犯规数',
    hc              SMALLINT       COMMENT '主队角球数',
    ac              SMALLINT       COMMENT '客队角球数',
    hy              SMALLINT       COMMENT '主队黄牌数',
    ay              SMALLINT       COMMENT '客队黄牌数',
    hr              SMALLINT       COMMENT '主队红牌数',
    ar              SMALLINT       COMMENT '客队红牌数',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_match (match_date, home_team, away_team),
    INDEX idx_season (season),
    INDEX idx_div (div),
    INDEX idx_home_team (home_team),
    INDEX idx_away_team (away_team)
) ENGINE=InnoDB COMMENT='比赛记录表';

-- ============================================
-- 2. 赔率数据表
-- ============================================
CREATE TABLE IF NOT EXISTS odds (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    match_id        BIGINT         COMMENT '关联比赛ID',
    -- Bet365 赔率
    b365h           DECIMAL(6,2)   COMMENT 'Bet365 主胜赔率',
    b365d           DECIMAL(6,2)   COMMENT 'Bet365 平局赔率',
    b365a           DECIMAL(6,2)   COMMENT 'Bet365 客胜赔率',
    -- Pinnacle 赔率
    psh             DECIMAL(6,2)   COMMENT 'Pinnacle 主胜赔率',
    psd             DECIMAL(6,2)   COMMENT 'Pinnacle 平局赔率',
    psa             DECIMAL(6,2)   COMMENT 'Pinnacle 客胜赔率',
    -- 大小球
    b365_over25     DECIMAL(6,2)   COMMENT 'Bet365 大2.5球赔率',
    b365_under25    DECIMAL(6,2)   COMMENT 'Bet365 小2.5球赔率',
    -- 亚盘
    ahh             DECIMAL(4,2)   COMMENT '亚盘让球数',
    b365ahh         DECIMAL(6,2)   COMMENT 'Bet365 亚盘主队赔率',
    b365aha         DECIMAL(6,2)   COMMENT 'Bet365 亚盘客队赔率',
    -- 赔率时间戳（实时赔率用）
    odds_timestamp  TIMESTAMP      COMMENT '赔率采集时间',
    is_realtime     TINYINT(1) DEFAULT 0 COMMENT '是否为实时赔率',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(id),
    INDEX idx_match_id (match_id)
) ENGINE=InnoDB COMMENT='赔率数据表';

-- ============================================
-- 3. 预测记录表
-- ============================================
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    match_id        BIGINT         COMMENT '关联比赛ID',
    user_id         BIGINT         COMMENT '请求用户ID',
    predict_type    VARCHAR(20)    COMMENT '预测类型 (advance=提前预测, realtime=实时预测)',
    predict_ftr     CHAR(1)        COMMENT '预测结果 (H/D/A)',
    predict_fthg    SMALLINT       COMMENT '预测主队进球',
    predict_ftag    SMALLINT       COMMENT '预测客队进球',
    confidence      DECIMAL(5,2)   COMMENT '置信度 (0-100)',
    reasoning       TEXT           COMMENT 'Agent 推理过程',
    actual_ftr      CHAR(1)        COMMENT '实际结果（赛后回填）',
    is_correct      TINYINT(1)     COMMENT '是否预测正确（赛后回填）',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_match_id (match_id),
    INDEX idx_user_id (user_id),
    INDEX idx_predict_type (predict_type)
) ENGINE=InnoDB COMMENT='预测记录表';

-- ============================================
-- 4. 用户表
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    username        VARCHAR(50) NOT NULL UNIQUE,
    email           VARCHAR(100),
    hashed_password VARCHAR(255) NOT NULL,
    is_active       TINYINT(1) DEFAULT 1,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB COMMENT='用户表';

-- ============================================
-- 5. 会话记录表
-- ============================================
CREATE TABLE IF NOT EXISTS conversations (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id         BIGINT,
    title           VARCHAR(200)   COMMENT '会话标题',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB COMMENT='会话记录表';

-- ============================================
-- 6. 消息记录表
-- ============================================
CREATE TABLE IF NOT EXISTS messages (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    conversation_id BIGINT,
    role            VARCHAR(20)    COMMENT '角色 (user/assistant/system)',
    content         TEXT           COMMENT '消息内容',
    agent_type      VARCHAR(50)    COMMENT '处理的Agent类型',
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id),
    INDEX idx_conversation_id (conversation_id)
) ENGINE=InnoDB COMMENT='消息记录表';

