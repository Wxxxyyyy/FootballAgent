// ⚽ Football Agent - Neo4j 图谱 Schema 定义
// 执行方式: 在 Neo4j Browser 中逐条执行，或通过 init_neo4j.py 脚本执行

// ============================================
// 1. 约束和索引
// ============================================

// 球队名称唯一约束
CREATE CONSTRAINT team_name_unique IF NOT EXISTS
FOR (t:Team) REQUIRE t.name IS UNIQUE;

// 联赛名称唯一约束
CREATE CONSTRAINT league_name_unique IF NOT EXISTS
FOR (l:League) REQUIRE l.name IS UNIQUE;

// 赛季唯一约束
CREATE CONSTRAINT season_name_unique IF NOT EXISTS
FOR (s:Season) REQUIRE s.name IS UNIQUE;

// 比赛ID唯一约束
CREATE CONSTRAINT match_id_unique IF NOT EXISTS
FOR (m:Match) REQUIRE m.match_id IS UNIQUE;

// ============================================
// 2. 节点标签说明
// ============================================
// (:Team)    - 球队节点: {name, country, city, founded_year}
// (:League)  - 联赛节点: {name, country, code}  (如 "Premier League", code="E0")
// (:Season)  - 赛季节点: {name}  (如 "2024-2025")
// (:Match)   - 比赛节点: {match_id, date, time, fthg, ftag, ftr, hthg, htag, htr}

// ============================================
// 3. 关系说明
// ============================================
// (Team)-[:PLAYS_IN]->(League)           球队属于联赛
// (Team)-[:HOME_MATCH]->(Match)          主场比赛
// (Team)-[:AWAY_MATCH]->(Match)          客场比赛
// (Match)-[:IN_SEASON]->(Season)         比赛所属赛季
// (Match)-[:IN_LEAGUE]->(League)         比赛所属联赛
// (Team)-[:RIVAL_OF]->(Team)             德比/宿敌关系

