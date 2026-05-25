"""文件上传接口模块"""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import config
from app.services.vector_index_service import vector_index_service

router = APIRouter()

# 从配置读取
ALLOWED_EXTENSIONS = [ext.strip().lower() for ext in config.allowed_upload_extensions.split(",") if ext.strip()]
MAX_FILE_SIZE = config.max_upload_size_mb * 1024 * 1024
UPLOAD_DIR = Path(config.upload_dir)

# 危险扩展名黑名单（始终禁止，即使不在白名单中）
DANGEROUS_EXTENSIONS = {
    ".exe", ".sh", ".bat", ".cmd", ".ps1", ".py", ".js", ".vbs", ".wsf",
    ".msi", ".com", ".scr", ".pif", ".jar", ".app", ".bin",
}


def _safe_filename(filename: str) -> str:
    """生成安全文件名，防止路径穿越和特殊字符

    策略：
    1. 只保留文件名部分（去掉路径）
    2. 去除危险字符
    3. 添加 UUID 前缀防冲突
    """
    # 只取文件名，去掉任何路径部分
    basename = os.path.basename(filename)

    # 去掉所有特殊字符，只保留字母数字、点、连字符、下划线
    safe = ""
    for char in basename:
        if char.isalnum() or char in (".", "-", "_"):
            safe += char
        else:
            safe += "_"

    # 如果清理后为空，生成随机名
    if not safe or safe.startswith("."):
        ext = os.path.splitext(basename)[1] if "." in basename else ""
        safe = f"upload_{uuid.uuid4().hex[:8]}{ext}"

    # 添加 UUID 前缀防止碰撞
    name, ext = os.path.splitext(safe)
    return f"{name}_{uuid.uuid4().hex[:8]}{ext}"


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件并自动创建向量索引

    - 校验文件大小（MAX_UPLOAD_SIZE_MB 配置项）
    - 校验扩展名白名单（ALLOWED_UPLOAD_EXTENSIONS）
    - 防止路径穿越
    - 禁止可执行文件
    """
    try:
        # 1. 验证文件名
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        # 2. 获取扩展名
        file_extension = os.path.splitext(file.filename)[1].lower()

        # 3. 检查扩展名白名单
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式 '{file_extension}'，仅支持: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # 4. 禁止可执行文件（双重保障）
        if file_extension in DANGEROUS_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"禁止上传可执行文件: '{file_extension}'",
            )

        # 5. 读取文件内容并校验大小
        content = await file.read()

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件大小 ({len(content)} 字节) 超过限制 ({MAX_FILE_SIZE} 字节，约 {config.max_upload_size_mb}MB)",
            )

        # 6. 生成安全文件名
        safe_filename = _safe_filename(file.filename)

        # 7. 创建上传目录
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 8. 验证最终路径在 UPLOAD_DIR 下（防止路径穿越）
        file_path = (UPLOAD_DIR / safe_filename).resolve()
        upload_dir_resolved = UPLOAD_DIR.resolve()
        if not str(file_path).startswith(str(upload_dir_resolved)):
            raise HTTPException(status_code=400, detail="非法的文件路径")

        # 9. 保存文件
        file_path.write_bytes(content)
        logger.info(f"文件上传成功: {file_path}")

        # 10. 自动创建向量索引
        try:
            logger.info(f"开始为上传文件创建向量索引: {file_path}")
            vector_index_service.index_single_file(str(file_path))
            logger.info(f"向量索引创建成功: {file_path}")
        except Exception as e:
            logger.error(f"向量索引创建失败: {file_path}, 错误: {e}")

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success",
                "data": {
                    "filename": safe_filename,
                    "file_path": str(file_path),
                    "size": len(content),
                },
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}") from e


@router.post("/index_directory")
async def index_directory(directory_path: str = None):
    """索引指定目录下的所有文件"""
    try:
        target = directory_path or str(UPLOAD_DIR)
        logger.info(f"开始索引目录: {target}")
        result = vector_index_service.index_directory(target)
        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success" if result.success else "partial_success",
                "data": result.to_dict(),
            },
        )
    except Exception as e:
        logger.error(f"索引目录失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引目录失败: {e}") from e
