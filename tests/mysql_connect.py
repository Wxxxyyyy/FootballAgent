import os
import pymysql
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

def test_connection():
    print("⏳ 正在尝试连接 MySQL...")
    try:
        # 建立数据库连接
        connection = pymysql.connect(
            host=os.getenv("MYSQL_HOST"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE"),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5 # 设置 5 秒超时
        )
        
        print("✅ 成功连接到 MySQL 数据库！\n")
        
        # 顺便查一下你的 match_master 表，看看数据在不在
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION() as version;")
            version = cursor.fetchone()
            print(f"🖥️  MySQL 版本: {version['version']}")
            
            cursor.execute("SELECT COUNT(*) as count FROM match_master;")
            count = cursor.fetchone()
            print(f"📊 表 `match_master` 数据量: {count['count']} 条")
            
        # 关闭连接
        connection.close()
        print("\n🎉 测试完美通过，数据库随时待命！")
        
    except pymysql.MySQLError as e:
        print(f"\n❌ 数据库报错了 (MySQLError): {e}")
    except Exception as e:
        print(f"\n❌ 发生了其他错误: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_connection()