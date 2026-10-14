# -*- coding: utf-8 -*-
from PIL import Image
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from typing import Dict, List, Tuple, Any
import math
import os
import random
import re
import sys
import time

try:
    from utils import logger
except ImportError:
    from agent.utils import logger

try:
    from utils import SendJinSan
except ImportError:
    from agent.utils import SendJinSan

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

try:
    from .searchAnswer import SearchQuestions, format_answer
except (ImportError, ValueError):
    try:
        from searchAnswer import SearchQuestions, format_answer
    except ImportError:
        try:
            from agent.custom.recognition.searchAnswer import SearchQuestions, format_answer
        except ImportError:
            _cur_dir = os.path.dirname(os.path.abspath(__file__))
            if _cur_dir not in sys.path:
                sys.path.insert(0, _cur_dir)
            from searchAnswer import SearchQuestions, format_answer

# 三界奇缘第一选项区域 (x, y, w, h)，参考 sanjieqiyuan.json 中定义
SJQY_FIRST_OPTION_BOX = [450, 340, 227, 92]


def calculate_reading_delay(text: str = "", options: Any = None) -> float:
    """
    计算基于字符长度的对数正态/拟人认知延时：
    认知延时 = 基础阅读时延 0.6s + 题目/选项字符数 * 0.05s + 随机浮动 0.3~0.8s
    模拟真人读题审题认知过程。
    """
    char_count = len(text.strip()) if text else 0
    if options:
        if isinstance(options, dict):
            for v in options.values():
                if v:
                    char_count += len(str(v).strip())
        elif isinstance(options, (list, tuple, set)):
            for v in options:
                if v:
                    char_count += len(str(v).strip())
        elif isinstance(options, str):
            char_count += len(options.strip())

    base_delay = 0.6
    char_delay = char_count * 0.05
    # 随机浮动 0.3~0.8s（对数正态分布截断在 [0.3, 0.8]）
    try:
        raw_jitter = random.lognormvariate(math.log(0.51), 0.25)
        random_jitter = max(0.3, min(0.8, raw_jitter))
    except Exception:
        random_jitter = random.uniform(0.3, 0.8)

    return base_delay + char_delay + random_jitter


@AgentServer.custom_recognition("reco2")
class reco2_sjqy(CustomRecognition):
    def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
        logger.info("进入reco2_sjqy")
        return CustomRecognition.AnalyzeResult(box=(6,134,57,46),detail="福利")



@AgentServer.custom_recognition("sjqy_tiku_V3")
class sjqy_tiku_V3(CustomRecognition):

    def analyze(
         self,
         context: Context,
         argv: CustomRecognition.AnalyzeArg,
     ) -> CustomRecognition.AnalyzeResult:
        # logger.info("进入三界奇缘答题agent_V3")

            # 根据位置排序OCR结果
        def sort_ocr_results_by_position(ocr_results):
            # 定义行高阈值，如果两个框的y坐标差距小于这个值，认为它们在同一行
            row_height_threshold = 20
            
            # 按y坐标分组（将接近的y坐标视为同一行）
            rows = {}
            for result in ocr_results:
                y = result.box[1]  # y坐标
                assigned = False
                
                # 检查是否可以分配到现有行
                for row_y in rows.keys():
                    if abs(y - row_y) < row_height_threshold:
                        rows[row_y].append(result)
                        assigned = True
                        break
                
                # 如果不能分配到现有行，创建新行
                if not assigned:
                    rows[y] = [result]
            
            # 对每一行内的元素按x坐标排序
            for row_y in rows:
                rows[row_y].sort(key=lambda r: r.box[0])
            
            # 按y坐标对行进行排序，并将所有结果合并到一个列表中
            sorted_results = []
            for row_y in sorted(rows.keys()):
                sorted_results.extend(rows[row_y])
            
            return sorted_results

        # 未找到答案次数
        NotAnswerCount = 0
        # 当第一次进入答题时，需要暂停一秒后在开始截图。
        max_hit = context.get_hit_count("活动-三界奇缘-开始答题_agent")
        if max_hit == 0:
            time.sleep(1.5)
        
        #识别三界奇缘题目
        image1 = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition(
            "三界奇缘题目",
            image1,
            pipeline_override={"三界奇缘题目": {"roi" : [447,40,673,94],
                                                "expected":[".*"],
                                                "recognition": "OCR",
                                                "replace":[
                                                    ["味","昧"],
                                                    ["邮","邺"],
                                                    ["尺","尸"],
                                                    ["频","濒"],
                                                    ["铜","锢"]
                                                ]
                                                }
                                }
            )
        #识别题目返回值
        # logger.info("reco_detail为：{reco_detail}")
        
        # logger.info(reco_detail.box)
        # 没有识别到题目
        if not reco_detail or not reco_detail.hit:
            logger.info("没有识别到题目")
            # logger.info(f"未在题库中搜索到答案次数:{NotAnswerCount}，请反馈开发者填充题库。")
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="答题结束")
        filtered_results= reco_detail.filtered_results
        #按照box进行排序
        sorted_results= sort_ocr_results_by_position(filtered_results)
        #整合题目
        ext=''
        for item in sorted_results:
            if item.text!='':
                ext+=item.text
        pattern = r'第\d+题：|[（(]\d+/\d+[）)]'
        def clean_string(s):
            return re.sub(pattern, '', s).strip()
        text = clean_string(ext)
        # 获取答案列表（供 OCR expected 逐个匹配）
        answer_list, confidence, match_type = SearchQuestions(text)
        # 展示用字符串（日志 / 金山文档），不用于 OCR expected
        results_value = format_answer(answer_list)
        # 获取时间并格式化
        formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 如果可信度为零，点击第一个答案
        if confidence < 80:
            # logger.info(f"题库中未找到答案，问题为:{text}.请反馈开发者填充题库")
            # 预留接口，未在题库中找得到问题，可以把截图发送到服务器。
            # context.tasker.controller.post_screencap().wait().save("{text}.png")
            # 发送到金山文档1.识别到的问题，2.匹配度，3.题库答案，4.时间
            data_to_send = [text,confidence,results_value,formatted]
            # logger.info(f"发送到金山文档的数据为:{data_to_send}")   
            if SendJinSan.send(data_to_send):
                logger.info(f"[color:red]未与题库匹配。[/color]识别题目：{text}，匹配度：{confidence}，已经登记在线文档")
            else:
                logger.error(f"登记在线文档失败:{text}")

            # 基于字符长度的拟人认知阅读延时
            reading_delay = calculate_reading_delay(text=text, options=answer_list)
            time.sleep(reading_delay)

            # 盲猜兜底：获取第一选项区域高斯随机落点并接入单人注意力调度器
            click_x, click_y = get_human_click_point(SJQY_FIRST_OPTION_BOX)
            try:
                instance_id = get_instance_id(context)
            except Exception:
                instance_id = "default"

            with HumanAttentionCoordinator.acquire_focus(instance_id):
                human_touch_down_up(context.tasker.controller, click_x, click_y)

            logger.info(f"[color:red]未与题库匹配，盲猜点击第一选项。[/color]识别题目：{text}，匹配度：{confidence}，落点：({click_x}, {click_y})，认知耗时：{reading_delay:.2f}s")
            time.sleep(random.uniform(2.5, 3.0))
            return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="题库缺少本问题")
        elif confidence >=80 and confidence < 100:
            
            # 发送到金山文档
            data_to_send = [text,confidence,results_value,formatted]
            # logger.info(f"发送到金山文档的数据为:{data_to_send}")   
            if SendJinSan.send(data_to_send):
                logger.info(f"本问题与题库非100%匹配:{text},已经登记在线文档")
            else:
                logger.error(f"登记在线文档失败:{text}")
        # 识别答案位置
        # new_context = context.clone()
        # image2 = context.tasker.controller.post_screencap().wait().get()
        new_reco_detail = context.run_recognition(
                        "三界奇缘答案位置",
                        image1,
                        pipeline_override={"三界奇缘答案位置": {"roi" : [439,218,678,212],
                                                            "expected":[re.escape(a) for a in answer_list],
                                                            "recognition": "OCR"
                                                            }
                                            }
                        )
        # logger.info("new_reco_detail为：{new_reco_detail}")
        # logger.info(new_reco_detail.box)
        # 点击答案
        if new_reco_detail and new_reco_detail.hit:
            box = new_reco_detail.box  # 假设box为(x, y, w, h)
            # 基于字符长度的拟人认知阅读延时
            reading_delay = calculate_reading_delay(text=text, options=answer_list)
            time.sleep(reading_delay)

            # 获取选项点击落点截断高斯随机落点并接入注意力互斥锁
            click_x, click_y = get_human_click_point(box)
            try:
                instance_id = get_instance_id(context)
            except Exception:
                instance_id = "default"

            with HumanAttentionCoordinator.acquire_focus(instance_id):
                human_touch_down_up(context.tasker.controller, click_x, click_y)

            # 等待点击操作完成
            # image2 = context.tasker.controller.post_screencap().wait().get()
            # logger.info(image2)
            logger.info(f"[color:blue]正确点击答案。[/color]识别题目：{text}。题库答案：{results_value}。匹配度：{confidence}。落点：({click_x}, {click_y})。认知耗时：{reading_delay:.2f}s")
            time.sleep(random.uniform(1.8, 2.2))
        else:#没找到答案，盲猜第一选项
            # 基于字符长度的拟人认知阅读延时
            reading_delay = calculate_reading_delay(text=text, options=answer_list)
            time.sleep(reading_delay)

            click_x, click_y = get_human_click_point(SJQY_FIRST_OPTION_BOX)
            try:
                instance_id = get_instance_id(context)
            except Exception:
                instance_id = "default"

            with HumanAttentionCoordinator.acquire_focus(instance_id):
                human_touch_down_up(context.tasker.controller, click_x, click_y)

            logger.info(f"[color:red]未在界面找到答案，盲猜点击第一选项。[/color]识别题目：{text}。题库答案：{results_value}。匹配度：{confidence}。落点：({click_x}, {click_y})。认知耗时：{reading_delay:.2f}s")
            time.sleep(random.uniform(1.0, 1.5))

        # logger.info(f"未在题库中搜索到答案次数:{NotAnswerCount}，请反馈开发者填充题库。")
        return CustomRecognition.AnalyzeResult(box=(0,0,0,0),detail="答题结束")
