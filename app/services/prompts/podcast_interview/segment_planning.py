#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: 播客访谈解说-片段规划
@File   : segment_planning.py
@Description: 播客访谈片段规划提示词
"""

from ..base import ParameterizedPrompt, PromptMetadata, ModelType, OutputFormat


class SegmentPlanningPrompt(ParameterizedPrompt):
    """播客访谈片段规划提示词"""

    def __init__(self):
        metadata = PromptMetadata(
            name="segment_planning",
            category="podcast_interview",
            version="v1.1",
            description="从访谈候选金句里选出最终片段，规划关系链与原声/解说节奏",
            model_type=ModelType.TEXT,
            output_format=OutputFormat.JSON,
            tags=["播客", "访谈", "片段规划", "关系链", "原声", "金句"],
            parameters=[
                "drama_name",
                "drama_genre",
                "plot_analysis",
                "subtitle_content",
                "narration_language",
            ],
        )
        super().__init__(
            metadata,
            required_parameters=["drama_name", "plot_analysis", "subtitle_content"],
        )

        self._system_prompt = (
            "你是一位播客访谈剪辑规划师。你的任务是从字幕中选择可剪辑的嘉宾原声片段，"
            "并规划它们之间的关系链。必须严格输出JSON，不能写解说文案，不能输出Markdown或额外说明。"
        )

    def get_template(self) -> str:
        return """# 播客访谈片段规划任务

## 目标
为访谈节目《${drama_name}》规划一组可直接剪辑的原声片段，并标注它们之间的关系链。
你只负责选片段和标注用途，不写最终解说台词。

## 访谈分析材料
<analysis>
${plot_analysis}
</analysis>

## 原始字幕（访谈逐字稿）
<subtitles>
${subtitle_content}
</subtitles>

## 解说台词目标语言
<narration_language>
${narration_language}
</narration_language>

## 访谈类型
<drama_genre>
${drama_genre}
</drama_genre>

## 叙事规划目标
你不是在挑精彩片段合集，而是在规划一条观众能顺着看懂的访谈解说线：
必须先把访谈分析里的"候选金句"排成一条关系链，再决定段数。

## 关系链规则（核心）

相邻两段必须且只能属于以下五种之一：

| 关系 | 判定问题 | 连接词示例 |
|---|---|---|
| 限定 | 后段是否改变了前段的结论（前段对但不全对）？ | 但问题在于 / 前提是 / 他自己也承认 / 反过来看 |
| 修正 | 后段是否改口、收回或修正了前段的说法？ | 说到这儿他停了一下 / 后来他修正了 / 其实也不全是这样 |
| 掀翻 | 后段是否否定了前段的前提或提问本身？ | 但真正有意思的是 / 他根本不认同这个问法 |
| 举例 | 后段是否是前段的具体例子或证据？ | 他举了个例子 / 这件事他专门讲了一遍 |
| 递进 | 后段是否未改变前段结论、但把论述推向更根本的原因？ | 更进一步 / 更根本的是 / 往深一层说 |

### 限定与递进的排他判据（必须二选一，不得并存）

- 先问：后段有没有改变前段的结论？
  - 改变了 → 限定
  - 没改变，只是把前段推向更根本的原因 → 递进
- 两者都判不出 → 降级为并列，按"并列最多出现一次"规则处理

连接词只作提示，最终以关系判定为准。连接词与判定冲突时，以判定为准。

### 关系链推导段数

1. 写出主线需要回答的那个追问。
2. 列出回答该追问所需的最少推进次数。每需要一次推进，增加一段。
3. 把关系链写成一行：A —关系→ B —关系→ C。这一行有几个节点，段数就是几。
4. 用容量校验：段数超出容量 → 砍掉关系链中最弱的一环；砍到撑不住主线 → 收窄主线。

段数是关系链的输出，不是输入。

### 关系链硬规则

- 相邻两段标不出五种关系中的任何一种 → 该段不得进链，段数相应减少。
- 并列最多出现一次（最多两段并排）。并列第二次出现 → 关系链已断，重推。
- 关系链减到撑不住主线 → 收窄主线，不要强行凑段数。

## 开场规则
- 第一个片段必须是 OST=1 嘉宾原声钩子，不能是主持人解说。
- 开场钩子优先选择访谈分析里"候选金句"中张力最强的原话。

## 规划规则
1. 只能使用原始字幕中真实存在的视频编号、视频文件名和时间范围。
2. timestamp 必须是对应 video_id 内部的局部时间戳，禁止换算成多个视频拼接后的累计时间。
3. 同一 video_id 内的片段不得交叉或重叠。
4. 每段原声优先保留完整金句（含限定条件和例外），不要为了缩短而破坏原话。
5. 原声片段单段优先控制在 5-15 秒；超过 20 秒考虑拆分（但要保留语义完整性）。
6. 解说片段（OST=0）用于接/补/点，单段承载一个信息点。
7. 访谈类内容的 OST 分布：
   - OST=1（纯原声）：55-65%（嘉宾说话是核心价值）
   - OST=0（纯解说）：25-35%（主持人串场、观点提炼）
   - OST=2（解说+原声）：5-10%（过渡段）
8. 跨 video_id 切换前后必须至少有一个 OST=0 片段作为剧情桥段。
9. 结尾优先停在原声（最后一段 OST=1 播完直接结束），或用一个开放问题收住。
10. 访谈类型由用户手动选择为 ${drama_genre}，不得自行改判；选片段时优先服务该类型的核心看点。
11. 容量换算：每 3-4 分钟成片容得下 1 个原声片段及其间的解说。按此换算，得到本片段的段数上限。

## 输出格式
只输出严格 JSON：

{
  "segments": [
    {
      "_id": 1,
      "video_id": 1,
      "video_name": "1.mp4",
      "timestamp": "00:05:23,000-00:05:41,000",
      "OST": 1,
      "story_role": "开场钩子",
      "relation_to_next": "限定",
      "intent": "用嘉宾反直觉判断开场，制造'钱多怎么会是风险'的追问"
    }
  ]
}

字段说明：
- story_role：开场钩子 / 核心论断 / 举证 / 限定 / 转折 / 收束
- relation_to_next：限定 / 修正 / 掀翻 / 举例 / 递进 / null（最后一段填 null）
- intent：一句话说明这段在原声叙事中的作用

现在请规划访谈节目《${drama_name}》的解说片段。"""