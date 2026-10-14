# -*- coding: utf-8 -*-
"""底层拟人触控核心模块 (Human Touch Core)

提供符合真实人类生理操作特征的落点生成与触控动作：
1. 二维截断高斯落点算法 (get_human_click_point)：均值在识别区域中心，截断在内边距 margin_ratio 内，
   消除绝对几何中心点击（零方差）特征并防止点击溢出边界。
2. 拟人按下抬起动态 (human_touch_down_up)：保持 65~125ms 指腹按压时长 (Dwell Time)，40% 概率触发 1px 肌肉微颤。
3. 拟人点击组合接口 (human_click)：整合落点生成与按下抬起。
4. 拟人滑动轨迹 (human_swipe)：基于三阶贝塞尔手势弧线与 SmoothStep S 型速度平滑。
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import Any, Tuple, Union

# 确保 agent 目录在 sys.path 中，兼容 package 导入与模块直接导入
_current_dir = os.path.dirname(os.path.abspath(__file__))  # agent/utils
_agent_dir = os.path.dirname(_current_dir)                 # agent
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

# 兼容 package 导入别名（agent.utils.human_touch 与 utils.human_touch）
_this_module = sys.modules[__name__]
if "agent.utils.human_touch" not in sys.modules:
    sys.modules["agent.utils.human_touch"] = _this_module
if "utils.human_touch" not in sys.modules:
    sys.modules["utils.human_touch"] = _this_module

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

            logger = logging.getLogger("HumanTouch")


def _safe_wait(job: Any) -> None:
    """若操作返回 MAA 异步任务对象，调用 wait() 等待其完成"""
    if job is not None and hasattr(job, "wait") and callable(getattr(job, "wait")):
        job.wait()


def _unwrap_controller(controller: Any) -> Any:
    """支持直接传入 Context 或 Tasker，自动解包出底层 Controller"""
    if controller is not None:
        if hasattr(controller, "tasker") and hasattr(controller.tasker, "controller"):
            return controller.tasker.controller
        if hasattr(controller, "controller"):
            return controller.controller
    return controller


def _parse_box(box: Any) -> Tuple[float, float, float, float]:
    """
    解析各种格式的检测框坐标为 (x, y, w, h)。
    支持格式：
    - [x, y, w, h] / (x, y, w, h)
    - [x, y] / (x, y) 点坐标（视作 w=0, h=0）
    - 具有 .box 属性的对象（如 MAA AnalyzeResult, OCRResult, RecoDetail）
    - 具有 x, y, w, h 或 width, height 属性的 Rect 结构体
    - 包含 x, y, w, h 或 x1, y1, x2, y2 的字典
    """
    if box is None:
        return 0.0, 0.0, 0.0, 0.0

    # 1. 若对象带有 .box 属性（MAA 识别结果）
    if hasattr(box, "box") and not callable(getattr(box, "box")):
        sub_box = getattr(box, "box")
        if sub_box is not None:
            return _parse_box(sub_box)

    # 2. 若对象带有 x, y 坐标属性（Rect 类对象）
    if hasattr(box, "x") and hasattr(box, "y"):
        x = float(getattr(box, "x", 0.0))
        y = float(getattr(box, "y", 0.0))
        w = float(getattr(box, "w", getattr(box, "width", 0.0)))
        h = float(getattr(box, "h", getattr(box, "height", 0.0)))
        return x, y, w, h

    # 3. 若为字典类型
    if isinstance(box, dict):
        if "x" in box and "y" in box:
            x = float(box["x"])
            y = float(box["y"])
            w = float(box.get("w", box.get("width", 0.0)))
            h = float(box.get("h", box.get("height", 0.0)))
            return x, y, w, h
        if "x1" in box and "x2" in box:
            x1 = float(box["x1"])
            y1 = float(box.get("y1", 0.0))
            x2 = float(box["x2"])
            y2 = float(box.get("y2", 0.0))
            return min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)

    # 4. 若为列表或元组
    if isinstance(box, (list, tuple)):
        if len(box) >= 4:
            x = float(box[0])
            y = float(box[1])
            w = float(box[2])
            h = float(box[3])
            # 兼容负宽高修正
            if w < 0:
                x, w = x + w, -w
            if h < 0:
                y, h = y + h, -h
            return x, y, w, h
        elif len(box) >= 2:
            return float(box[0]), float(box[1]), 0.0, 0.0

    return 0.0, 0.0, 0.0, 0.0


def get_human_click_point(box: list, margin_ratio: float = 0.15) -> tuple[int, int]:
    """
    在目标识别框内按二维截断高斯分布生成随机落点。
    均值在区域中心，截断在内边距 margin_ratio 内，防止点击绝对几何中心（零方差）或溢出边界。

    :param box: [x, y, w, h]、识别结果对象或检测框坐标
    :param margin_ratio: 安全内边距比例，默认 0.15 (15%)
    :return: (int(x), int(y)) 整型落点坐标
    """
    x, y, w, h = _parse_box(box)

    # 宽高非正时，直接返回基准坐标整型值
    if w <= 0.0 or h <= 0.0:
        return int(round(x)), int(round(y))

    # 安全内边距（至少 2px）
    safe_margin_x = max(2.0, w * margin_ratio)
    safe_margin_y = max(2.0, h * margin_ratio)

    # 若框体过小导致内边距超过半宽/半高，回退到按 10% 比例收缩
    if safe_margin_x * 2.0 >= w:
        safe_margin_x = max(0.0, w * 0.1)
    if safe_margin_y * 2.0 >= h:
        safe_margin_y = max(0.0, h * 0.1)

    min_x, max_x = x + safe_margin_x, x + w - safe_margin_x
    min_y, max_y = y + safe_margin_y, y + h - safe_margin_y

    if min_x > max_x:
        min_x, max_x = x, x + w
    if min_y > max_y:
        min_y, max_y = y, y + h

    center_x = x + w / 2.0
    center_y = y + h / 2.0

    # 3-sigma 覆盖截断区间，使高斯采样集中在中心偏离区域，极少超出 safe_margin
    sigma_x = max(1.0, (max_x - min_x) / 6.0)
    sigma_y = max(1.0, (max_y - min_y) / 6.0)

    # 截断高斯采样（最多 10 次）
    for _ in range(10):
        rx = random.gauss(center_x, sigma_x)
        ry = random.gauss(center_y, sigma_y)
        if min_x <= rx <= max_x and min_y <= ry <= max_y:
            ret_x = int(round(rx))
            ret_y = int(round(ry))
            # 避免恰好落在整数绝对几何中心（若区间有调整余地）
            if ret_x == int(round(center_x)) and ret_y == int(round(center_y)):
                cand_x = ret_x + random.choice([-1, 1])
                if min_x <= cand_x <= max_x:
                    ret_x = cand_x
                else:
                    cand_y = ret_y + random.choice([-1, 1])
                    if min_y <= cand_y <= max_y:
                        ret_y = cand_y
            return ret_x, ret_y

    # 极端采样兜底
    return int(round(center_x)), int(round(center_y))


def human_touch_down_up(controller: Any, target_x: int, target_y: int) -> None:
    """
    模拟真实人类指腹接触屏幕与按压过程：
    1. post_touch_down 按下
    2. 指腹停留时长 (Dwell Time) 在 65ms ~ 125ms 之间随机分布
    3. 40% 概率触发按压期间的 1 像素肌肉微颤抖/轻微位移 (post_touch_move)
    4. post_touch_up 抬起（try...finally 保证必抬起）

    :param controller: MAA 控制器实例（或 Context / Tasker）
    :param target_x: 目标 X 坐标
    :param target_y: 目标 Y 坐标
    """
    ctrl = _unwrap_controller(controller)
    tx = int(round(target_x))
    ty = int(round(target_y))

    # 按压时长 65~125ms（随机分布）
    dwell_time = random.uniform(0.065, 0.125)

    if hasattr(ctrl, "post_touch_down") and callable(getattr(ctrl, "post_touch_down")):
        _safe_wait(ctrl.post_touch_down(tx, ty))
        try:
            # 40% 概率产生 1px 微颤/轻微位移
            if random.random() < 0.4:
                half_dwell = dwell_time * 0.5
                time.sleep(half_dwell)

                # 产生 1 像素真实微动（8 个相邻像素方向随机选择其一）
                dx, dy = random.choice([
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)
                ])
                jx = tx + dx
                jy = ty + dy

                if hasattr(ctrl, "post_touch_move") and callable(getattr(ctrl, "post_touch_move")):
                    _safe_wait(ctrl.post_touch_move(jx, jy))

                time.sleep(half_dwell)
            else:
                time.sleep(dwell_time)
        finally:
            if hasattr(ctrl, "post_touch_up") and callable(getattr(ctrl, "post_touch_up")):
                _safe_wait(ctrl.post_touch_up())
    elif hasattr(ctrl, "post_click") and callable(getattr(ctrl, "post_click")):
        # 降级：仅支持 post_click 时
        time.sleep(random.uniform(0.02, 0.05))
        _safe_wait(ctrl.post_click(tx, ty))
    elif hasattr(ctrl, "click") and callable(getattr(ctrl, "click")):
        time.sleep(random.uniform(0.02, 0.05))
        _safe_wait(ctrl.click(tx, ty))
    else:
        raise AttributeError(
            f"Controller {type(ctrl)} does not support touch or click operations"
        )


def human_click(
    controller: Any,
    box: Any,
    pre_wait: float = 0.0,
    post_wait: float = 0.0,
    margin_ratio: float = 0.15,
) -> tuple[int, int]:
    """
    拟人化点击指定区域：生成截断高斯落点后执行拟人按下与抬起。

    :param controller: MAA 控制器实例（或 Context）
    :param box: [x, y, w, h]、识别结果对象或坐标列表
    :param pre_wait: 点击前等待时间（秒）
    :param post_wait: 点击后等待时间（秒）
    :param margin_ratio: 安全内边距比例，默认 0.15
    :return: (target_x, target_y) 实际点击的落点坐标
    """
    if pre_wait > 0.0:
        time.sleep(pre_wait)

    target_x, target_y = get_human_click_point(box, margin_ratio=margin_ratio)
    human_touch_down_up(controller, target_x, target_y)

    if post_wait > 0.0:
        time.sleep(post_wait)

    return target_x, target_y


def _smoothstep(t: float) -> float:
    """S 型速度曲线平滑函数 (3t^2 - 2t^3)，用于模拟人手先加速后减速过程"""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def human_swipe(
    controller: Any,
    start_box: Any,
    end_box: Any,
    duration_ms: int = 500,
    margin_ratio: float = 0.15,
    steps: int = 25,
) -> None:
    """
    拟人化滑动：三阶贝塞尔弧形曲线 + SmoothStep S 型速度平滑。

    :param controller: MAA 控制器实例（或 Context）
    :param start_box: 起点区域 [x, y, w, h] 或坐标
    :param end_box: 终点区域 [x, y, w, h] 或坐标
    :param duration_ms: 滑动总时长（毫秒）
    :param margin_ratio: 起终点选点内边距比例
    :param steps: 插值采样步数
    """
    ctrl = _unwrap_controller(controller)
    p0_x, p0_y = get_human_click_point(start_box, margin_ratio=margin_ratio)
    p3_x, p3_y = get_human_click_point(end_box, margin_ratio=margin_ratio)

    dx = p3_x - p0_x
    dy = p3_y - p0_y
    dist = math.hypot(dx, dy)

    if dist <= 0.0:
        human_touch_down_up(ctrl, p0_x, p0_y)
        return

    # 人手拇指滑动的自然弧度偏移（垂直于位移方向的单位法向量）
    nx = -dy / dist
    ny = dx / dist

    # 弧高：滑动距离的 5% ~ 15%，随机左弧或右弧
    arc_offset = dist * random.uniform(-0.15, 0.15)
    p1_x = p0_x + dx * 0.33 + nx * arc_offset * random.uniform(0.8, 1.2)
    p1_y = p0_y + dy * 0.33 + ny * arc_offset * random.uniform(0.8, 1.2)
    p2_x = p0_x + dx * 0.67 + nx * arc_offset * random.uniform(0.8, 1.2)
    p2_y = p0_y + dy * 0.67 + ny * arc_offset * random.uniform(0.8, 1.2)

    step_count = max(5, steps)
    dt = (duration_ms / 1000.0) / step_count

    if hasattr(ctrl, "post_touch_down") and callable(getattr(ctrl, "post_touch_down")):
        _safe_wait(ctrl.post_touch_down(p0_x, p0_y))
        try:
            time.sleep(random.uniform(0.015, 0.030))
            for i in range(1, step_count + 1):
                t = i / float(step_count)
                u = _smoothstep(t)

                # 三阶贝塞尔曲线公式:
                # B(u) = (1-u)^3*P0 + 3*(1-u)^2*u*P1 + 3*(1-u)*u^2*P2 + u^3*P3
                c0 = (1.0 - u) ** 3
                c1 = 3.0 * ((1.0 - u) ** 2) * u
                c2 = 3.0 * (1.0 - u) * (u ** 2)
                c3 = u ** 3

                cur_x = int(round(c0 * p0_x + c1 * p1_x + c2 * p2_x + c3 * p3_x))
                cur_y = int(round(c0 * p0_y + c1 * p1_y + c2 * p2_y + c3 * p3_y))

                if hasattr(ctrl, "post_touch_move") and callable(getattr(ctrl, "post_touch_move")):
                    _safe_wait(ctrl.post_touch_move(cur_x, cur_y))

                time.sleep(dt)
        finally:
            if hasattr(ctrl, "post_touch_up") and callable(getattr(ctrl, "post_touch_up")):
                _safe_wait(ctrl.post_touch_up())
    elif hasattr(ctrl, "post_swipe") and callable(getattr(ctrl, "post_swipe")):
        _safe_wait(ctrl.post_swipe(p0_x, p0_y, p3_x, p3_y, duration_ms))
    elif hasattr(ctrl, "swipe") and callable(getattr(ctrl, "swipe")):
        _safe_wait(ctrl.swipe(p0_x, p0_y, p3_x, p3_y, duration_ms))
    else:
        raise AttributeError(
            f"Controller {type(ctrl)} does not support post_touch_down or post_swipe"
        )


__all__ = [
    "get_human_click_point",
    "human_touch_down_up",
    "human_click",
    "human_swipe",
]
