"""启动嵌入式 Milvus Lite 服务器（代替 Docker Milvus）"""
import os
import sys
import time

# 使用 milvus-lite 的底层 API 直接启动 gRPC 服务器在固定端口
from milvus_lite.adapter.grpc.server import start_server_in_thread

DATA_DIR = os.path.join(os.path.dirname(__file__), "volumes", "milvus_lite")
os.makedirs(DATA_DIR, exist_ok=True)

server, db, port = start_server_in_thread(
    data_dir=DATA_DIR,
    host="0.0.0.0",
    port=19530,
)

print(f"Milvus Lite server started on 0.0.0.0:{port}", flush=True)

try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("Shutting down Milvus Lite...")
    server.stop(grace=0)
    db.close()
    sys.exit(0)
