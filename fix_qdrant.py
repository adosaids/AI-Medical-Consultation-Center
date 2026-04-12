"""
修复 QdrantStorage 析构错误
在 main.py 最开始导入此模块
"""
import sys

# 在导入 CAMEL 之前，修补 QdrantStorage
_original_excepthook = sys.excepthook

def patched_excepthook(exc_type, exc_value, exc_traceback):
    """忽略 QdrantStorage 的析构错误"""
    if exc_type == KeyError:
        str_val = str(exc_value)
        if 'local_data' in str_val or '_local_path' in str_val:
            return  # 忽略这个错误
    _original_excepthook(exc_type, exc_value, exc_traceback)

sys.excepthook = patched_excepthook

# 尝试修补 QdrantStorage 的 __del__ 方法
try:
    from camel.storages.vectordb_storages.qdrant import QdrantStorage

    # 保存原始的 __del__
    if hasattr(QdrantStorage, '__del__'):
        _original_del = QdrantStorage.__del__

        def _safe_del(self):
            try:
                _original_del(self)
            except (KeyError, AttributeError):
                pass

        QdrantStorage.__del__ = _safe_del
except ImportError:
    pass

print("Qdrant 补丁已应用")
