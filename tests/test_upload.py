"""上传安全测试"""

import pytest


class TestUploadSecurity:
    """上传接口安全测试"""

    def test_allowed_extension_md(self):
        """txt/md 文件应该被允许"""
        from app.api.file import ALLOWED_EXTENSIONS
        assert ".md" in ALLOWED_EXTENSIONS
        assert ".txt" in ALLOWED_EXTENSIONS

    def test_disallowed_extensions(self):
        """exe/sh/bat 不在白名单中，且在危险扩展名黑名单中"""
        from app.api.file import ALLOWED_EXTENSIONS, DANGEROUS_EXTENSIONS
        assert ".exe" not in ALLOWED_EXTENSIONS
        assert ".sh" not in ALLOWED_EXTENSIONS
        assert ".bat" not in ALLOWED_EXTENSIONS
        assert ".py" not in ALLOWED_EXTENSIONS
        assert ".exe" in DANGEROUS_EXTENSIONS
        assert ".sh" in DANGEROUS_EXTENSIONS
        assert ".bat" in DANGEROUS_EXTENSIONS

    def test_path_traversal_sanitized(self):
        """路径穿越文件名被清理：不包含 .. 和原始路径"""
        from app.api.file import _safe_filename

        result = _safe_filename("../../../etc/passwd.md")
        assert ".." not in result
        # 清理后的文件名不应该包含路径部分
        assert "/" not in result
        assert result.endswith(".md")

    def test_path_traversal_abs_path(self):
        """绝对路径：结果不含原始路径分隔符"""
        from app.api.file import _safe_filename

        result = _safe_filename("/etc/passwd.txt")
        assert result.endswith(".txt")
        # 不应该包含 etc 或 passwd (因为 os.path.basename 会去掉路径)
        # 实际 basename("passwd.txt") = "passwd.txt", 然后加 UUID
        # 所以 passwd 仍然在里面, 但 /etc/ 被去掉了
        assert "/" not in result

    def test_special_chars_removed(self):
        """特殊字符被替换为下划线"""
        from app.api.file import _safe_filename

        result = _safe_filename("test<script>.md")
        assert "<" not in result
        assert ">" not in result

    def test_max_file_size_configured(self):
        """确认文件大小限制配置"""
        from app.config import config
        assert config.max_upload_size_mb > 0
        assert config.max_upload_size_mb <= 100

    def test_upload_dir_configured(self):
        """确认上传目录配置"""
        from app.config import config
        assert config.upload_dir
        assert len(config.upload_dir) > 0

    def test_safe_filename_preserves_extension(self):
        """安全文件名保留原始扩展名"""
        from app.api.file import _safe_filename

        result = _safe_filename("document.txt")
        assert result.endswith(".txt")

        result = _safe_filename("report.pdf")
        assert result.endswith(".pdf")

    def test_safe_filename_adds_unique_id(self):
        """安全文件名添加了唯一标识防冲突"""
        from app.api.file import _safe_filename

        r1 = _safe_filename("test.md")
        r2 = _safe_filename("test.md")
        # 同一文件名两次调用应该生成不同结果
        assert r1 != r2
