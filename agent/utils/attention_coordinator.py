# -*- coding: utf-8 -*-
"""跨进程单人注意力调度模块 (HumanAttentionCoordinator)

多开/多实例环境下，确保同一时刻只有一个实例在模拟人类交互（鼠标移动、点击、滑动）。
当焦点在不同实例之间切换时，强制维护 250ms ~ 550ms 的人类视线/鼠标跨窗口位移物理时延，
而同实例连续操作无需等待切换时延，从而防止多开并发点击被服务端识别为群控外挂。

实现机制参考 agent/utils/utils.py 中的跨进程文件锁设计，具备严格的异常安全与超时降级机制，
严禁死锁。
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from contextlib import contextmanager
from typing import Optional, Tuple

try:
    from .logger import logger
except (ImportError, ValueError):
    try:
        from utils.logger import logger
    except (ImportError, ValueError):
        try:
            from agent.utils.logger import logger
        except (ImportError, ValueError):
            import logging

            logger = logging.getLogger("HumanAttentionCoordinator")


# 线程锁，保证同进程内多线程对注意力锁调度的串行化
_ATTENTION_THREAD_LOCK = threading.RLock()

# 默认锁超时时间（秒），加锁超时后退化为无锁执行，避免死锁
DEFAULT_LOCK_TIMEOUT_SECONDS: float = 10.0

# 记录是否已打印过超时警告，防止高频刷屏
_LOCK_WARNED = False


def _warn_attention_lock_timeout(timeout: float) -> None:
    """提示注意力锁超时警告（仅提示一次避免刷屏）"""
    global _LOCK_WARNED
    if _LOCK_WARNED:
        return
    _LOCK_WARNED = True
    msg = (
        f"[HumanAttentionCoordinator] 等待注意力互斥锁超过 {timeout:g}s，"
        f"本次交互退化为无锁执行以避免死锁。"
    )
    try:
        logger.warning(msg)
    except Exception:
        print(msg)


@contextmanager
def _attention_file_lock(lock_path: str, timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS):
    """跨进程与跨线程注意力互斥锁。

    - 进程内：使用 threading.RLock 进行互斥，支持超时。
    - 跨进程：使用 OS 文件锁（Windows 下使用 msvcrt.locking 非阻塞自旋，POSIX 下使用 fcntl.flock 非阻塞自旋）。
    - 防死锁与安全降级：若超过 timeout 仍未能获取锁，则记录告警并退化为无锁执行。
    - 异常安全：使用 finally 块确保持有的线程锁与文件锁必定释放，句柄必定关闭。
    """
    deadline = time.time() + timeout

    thread_locked = _ATTENTION_THREAD_LOCK.acquire(timeout=timeout)
    if not thread_locked:
        _warn_attention_lock_timeout(timeout)
        yield False
        return

    handle = None
    locked = False
    try:
        try:
            handle = open(lock_path, "a+b")
        except OSError as e:
            try:
                logger.warning(f"[HumanAttentionCoordinator] 打开锁文件 {lock_path} 失败: {e}")
            except Exception:
                pass
            handle = None

        if handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0, os.SEEK_END)
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()

                    while True:
                        try:
                            handle.seek(0)
                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                            locked = True
                            break
                        except OSError:
                            if time.time() >= deadline:
                                break
                            time.sleep(0.005)
                else:
                    import fcntl

                    while True:
                        try:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            locked = True
                            break
                        except (BlockingIOError, OSError):
                            if time.time() >= deadline:
                                break
                            time.sleep(0.005)
            except Exception as e:
                try:
                    logger.warning(f"[HumanAttentionCoordinator] 获取文件锁异常: {e}")
                except Exception:
                    pass
                locked = False

            if not locked:
                _warn_attention_lock_timeout(timeout)

        yield locked
    finally:
        if handle is not None:
            try:
                if locked:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                handle.close()
            except OSError:
                pass
        _ATTENTION_THREAD_LOCK.release()

# 别名，保持与方案文档原型契约一致
_storage_lock = _attention_file_lock


class HumanAttentionCoordinator:
    """全局单人注意力调度器。

    确保多开/多实例间同一时刻只有一个焦点在模拟用户交互。
    跨实例切换时，强制维护 250~550ms 的人类视线/鼠标跨窗口位移物理时延。
    同实例连续操作无需等待该切换时延。
    """

    # 存储目录与文件路径
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", "data"))
    _LOCK_FILE = os.path.join(_DATA_DIR, "attention.lock")
    _LAST_ACTION_FILE = os.path.join(_DATA_DIR, "last_act.stamp")

    # 默认参数
    _LOCK_TIMEOUT: float = DEFAULT_LOCK_TIMEOUT_SECONDS
    _MIN_SWITCH_DELAY: float = 0.25  # 250ms
    _MAX_SWITCH_DELAY: float = 0.55  # 550ms

    @classmethod
    @contextmanager
    def acquire_focus(cls, instance_id: str, timeout: Optional[float] = None):
        """类上下文管理器：获取模拟人类注意力的全局焦点。

        用法:
            with HumanAttentionCoordinator.acquire_focus(instance_id):
                human_click(controller, box)

        Args:
            instance_id: 当前正在交互的模拟器或账号唯一标识（如 adb_serial）。
            timeout: 获取锁的最大等待超时时间（秒），默认为 10.0 秒。
        """
        inst_id = str(instance_id) if instance_id is not None else "default"
        effective_timeout = timeout if timeout is not None else cls._LOCK_TIMEOUT

        # 确保数据目录存在
        lock_dir = os.path.dirname(cls._LOCK_FILE)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)

        with _attention_file_lock(cls._LOCK_FILE, timeout=effective_timeout):
            # 1. 检测跨实例切换，必要时注入 250ms ~ 550ms 视线/位移物理时延
            cls._apply_switch_delay_if_needed(inst_id)

            try:
                # 2. 进入业务操作
                yield inst_id
            finally:
                # 3. 退出时更新最后动作时间戳，确保即使业务抛出异常也正确维护时间状态
                cls._update_last_action(inst_id)

    # 别名兼容方案中的 AttentionCoordinator.acquire 命名
    acquire = acquire_focus

    def __init__(self, instance_id: str, timeout: Optional[float] = None):
        """支持实例上下文管理器模式:
            with HumanAttentionCoordinator(instance_id):
                ...
        """
        self.instance_id = instance_id
        self.timeout = timeout
        self._cm = None

    def __enter__(self) -> str:
        self._cm = self.acquire_focus(self.instance_id, timeout=self.timeout)
        return self._cm.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cm:
            return self._cm.__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def _apply_switch_delay_if_needed(cls, instance_id: str) -> float:
        """检查上一次动作记录。若检测到来自其他实例，强制注入 250ms ~ 550ms 的位移时延。
        同实例连续操作无需等待该切换时延。
        返回实际等待的秒数（0.0 表示未发生切换或无需等待）。
        """
        if not os.path.exists(cls._LAST_ACTION_FILE):
            return 0.0

        try:
            with open(cls._LAST_ACTION_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return 0.0

            parts = content.split("|", 1)
            if len(parts) != 2:
                return 0.0

            last_inst, last_time_str = parts[0], parts[1]
            last_time = float(last_time_str)

            # 跨实例切换检测
            if last_inst != instance_id:
                now = time.time()
                elapsed = max(0.0, now - last_time)
                required_delay = random.uniform(cls._MIN_SWITCH_DELAY, cls._MAX_SWITCH_DELAY)
                if elapsed < required_delay:
                    wait_time = required_delay - elapsed
                    try:
                        logger.debug(
                            f"[HumanAttention] 跨实例切换 ({last_inst} -> {instance_id})，"
                            f"注入视线/鼠标跨窗口位移时延: {wait_time * 1000:.1f}ms "
                            f"(要求时延: {required_delay * 1000:.1f}ms, 已流逝: {elapsed * 1000:.1f}ms)"
                        )
                    except Exception:
                        pass
                    time.sleep(wait_time)
                    return wait_time
        except Exception as e:
            try:
                logger.debug(f"[HumanAttention] 读取上一次动作记录异常: {e}")
            except Exception:
                pass

        return 0.0

    @classmethod
    def _update_last_action(cls, instance_id: str) -> None:
        """更新最后一次交互记录（实例 ID 与当前时间戳）。"""
        try:
            action_dir = os.path.dirname(cls._LAST_ACTION_FILE)
            if action_dir:
                os.makedirs(action_dir, exist_ok=True)

            record = f"{instance_id}|{time.time():.6f}"
            temp_file = f"{cls._LAST_ACTION_FILE}.tmp.{os.getpid()}"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(record)
            os.replace(temp_file, cls._LAST_ACTION_FILE)
        except Exception:
            try:
                with open(cls._LAST_ACTION_FILE, "w", encoding="utf-8") as f:
                    f.write(f"{instance_id}|{time.time():.6f}")
            except Exception as e:
                try:
                    logger.warning(f"[HumanAttention] 写入最后动作时间戳失败: {e}")
                except Exception:
                    pass

    @classmethod
    def get_last_action_info(cls) -> Optional[Tuple[str, float]]:
        """获取上一次动作的实例 ID 与时间戳，若无记录或解析失败返回 None。"""
        if not os.path.exists(cls._LAST_ACTION_FILE):
            return None
        try:
            with open(cls._LAST_ACTION_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return None
            parts = content.split("|", 1)
            if len(parts) == 2:
                return parts[0], float(parts[1])
        except Exception:
            pass
        return None

    @classmethod
    def clear_state(cls) -> None:
        """清理锁与时间戳文件（主要用于测试与重置状态）。"""
        for path in (cls._LAST_ACTION_FILE, cls._LOCK_FILE):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


# 别名兼容
AttentionCoordinator = HumanAttentionCoordinator

__all__ = [
    "HumanAttentionCoordinator",
    "AttentionCoordinator",
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
]
