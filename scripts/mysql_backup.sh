#!/usr/bin/env bash
# MySQL 每日备份（systemd timer 每天 04:00 调用）
# 备份 football_agent 库到 backups/，gzip 压缩，保留最近 14 份。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/backups"
mkdir -p "${BACKUP_DIR}"

# 从 .env 读取密码（避免硬编码）
MYSQL_PASSWORD="$(grep -E '^MYSQL_PASSWORD=' "${PROJECT_ROOT}/.env" | cut -d= -f2-)"
MYSQL_DATABASE="$(grep -E '^MYSQL_DATABASE=' "${PROJECT_ROOT}/.env" | cut -d= -f2-)"
MYSQL_DATABASE="${MYSQL_DATABASE:-football_agent}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTFILE="${BACKUP_DIR}/${MYSQL_DATABASE}_${STAMP}.sql.gz"

docker exec football_mysql mysqldump \
    -uroot -p"${MYSQL_PASSWORD}" \
    --single-transaction --quick --routines \
    "${MYSQL_DATABASE}" | gzip > "${OUTFILE}"

echo "[backup] $(date '+%F %T') 备份完成: ${OUTFILE} ($(du -h "${OUTFILE}" | cut -f1))"

# 保留最近 14 份，删除更旧的
ls -1t "${BACKUP_DIR}"/${MYSQL_DATABASE}_*.sql.gz 2>/dev/null | tail -n +15 | xargs -r rm -f
