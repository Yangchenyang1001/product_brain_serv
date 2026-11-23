"""
切片主键生成

chunk_id 是平台侧的资源标识（对外接口路径直接携带），必须由应用生成且保持稳定，
因此切片集合不使用 Milvus AutoID：AutoID 在 upsert 时会重新生成主键，导致旧 ID 失效。

生成规则参考雪花算法：高位毫秒时间戳 + 节点标识 + 毫秒内自增序号，
保证单进程严格递增、多进程（不同 pid）之间也不易重复，且始终落在 INT64 正整数范围。
"""
import os
import threading
import time

# 时间起点取 2023-11-14，缩短数值位宽；时间戳左移 22 位后仍远小于 INT64 上限。
_EPOCH_MS = 1_700_000_000_000
_NODE_BITS = 10
_SEQ_BITS = 12
_MAX_SEQ = (1 << _SEQ_BITS) - 1

_lock = threading.Lock()
_last_ms = 0
_seq = 0
# 节点标识取进程号低位，避免多实例并发导入时在同一毫秒生成相同主键。
_node_id = os.getpid() & ((1 << _NODE_BITS) - 1)


def next_chunk_id() -> int:
    """生成下一个切片主键（INT64 正整数，严格递增）。"""
    global _last_ms, _seq
    with _lock:
        current_ms = int(time.time() * 1000)
        # 时钟回拨时沿用最后一次时间戳，用序号推进，保证主键不倒退。
        if current_ms <= _last_ms:
            current_ms = _last_ms
            _seq += 1
            # 同一毫秒序号用尽后借用下一毫秒，避免主键重复。
            if _seq > _MAX_SEQ:
                current_ms += 1
                _seq = 0
        else:
            _seq = 0
        _last_ms = current_ms
        return (
            ((current_ms - _EPOCH_MS) << (_NODE_BITS + _SEQ_BITS))
            | (_node_id << _SEQ_BITS)
            | _seq
        )
