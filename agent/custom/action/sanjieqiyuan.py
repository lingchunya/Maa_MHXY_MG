import os
import json
from datetime import datetime

from PIL import Image
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

try:
    from utils import logger
except ImportError:
    from agent.utils import logger

try:
    from utils.human_touch import get_human_click_point, human_touch_down_up
    from utils.attention_coordinator import HumanAttentionCoordinator
    from utils.utils import get_instance_id
except ImportError:
    try:
        from agent.utils.human_touch import get_human_click_point, human_touch_down_up
        from agent.utils.attention_coordinator import HumanAttentionCoordinator
        from agent.utils.utils import get_instance_id
    except ImportError:
        from utils import get_human_click_point, human_touch_down_up, HumanAttentionCoordinator
        from utils import get_instance_id

# 左上角返回/关闭按钮安全点击区域 (x, y, w, h)，以原硬编码 (34, 137) 为中心
CLOSE_BUTTON_BOX = [24, 127, 20, 20]


@AgentServer.custom_action("sanjie")
class sanjie(CustomAction):
    """
    三界奇缘答题，
        1.图片
        2.ocr
        3.搜索答案
        4.识别答案位置
        5.return 答案位置
    """
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:
        
        logger.info("进入sanjie")
        click_x, click_y = get_human_click_point(CLOSE_BUTTON_BOX)
        try:
            instance_id = get_instance_id(context)
        except Exception:
            instance_id = "default"

        with HumanAttentionCoordinator.acquire_focus(instance_id):
            human_touch_down_up(context.tasker.controller, click_x, click_y)

        logger.debug(f"[sanjie] 拟人点击返回按钮: ({click_x}, {click_y})")
        return CustomAction.RunResult(success=True)
