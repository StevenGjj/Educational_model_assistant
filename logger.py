import logging
import os
from datetime import datetime

# 日志目录创建
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# 控制台处理器（简洁格式）
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)

# 文件处理器（保留JSON/文本格式，用于排查问题）
file_handler = logging.FileHandler(
    os.path.join(log_dir, f"rag_{datetime.now().strftime('%Y%m%d')}.log"),
    encoding="utf-8"
)
file_handler.setFormatter(
    logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)

# 全局配置
logging.basicConfig(
    level=logging.INFO,  # 仅显示INFO/ERROR级别
    handlers=[console_handler, file_handler],
    force=True
)

logger = logging.getLogger("rag_app")

# 禁用第三方库日志
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)