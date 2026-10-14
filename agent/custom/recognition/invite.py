from maa.agent.agent_server import AgentServer
from maa.custom_recognition  import CustomRecognition
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

@AgentServer.custom_recognition("invite")
class invite(CustomRecognition):
    """
    邀请队员
    "attach": {
            "name": ["姓名1", "姓名2"]

        }
        name: 需要要的的队员名字列表。
        

    """
    def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
        # 识别搜索位置并点击
        image = context.tasker.controller.post_screencap().wait().get()
        reco_result = context.run_recognition(
            "搜索",
            image,
            pipeline_override={
                "搜索":{"roi" : [336,224,64,38],
                        "expected":["搜索"],
                        "recognition": "OCR"
                       }
                }
        )
        # 搜索按钮安全区域，优先使用识别到的 box，未命中时回退到默认 ROI [336, 224, 64, 38]
        target_box = [336, 224, 64, 38]
        if reco_result and getattr(reco_result, "all_results", None):
            for res in reco_result.all_results:
                if getattr(res, "box", None):
                    target_box = res.box
        elif reco_result and getattr(reco_result, "box", None):
            target_box = reco_result.box

        click_x, click_y = get_human_click_point(target_box)
        logger.debug(f"[invite] 拟人点击搜索位置: ({click_x}, {click_y})")
        try:
            instance_id = get_instance_id(context)
        except Exception:
            instance_id = "default"

        with HumanAttentionCoordinator.acquire_focus(instance_id):
            human_touch_down_up(context.tasker.controller, click_x, click_y)


       
        return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="活跃度小于50,任务结束")



