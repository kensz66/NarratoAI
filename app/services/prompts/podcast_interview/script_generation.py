#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: 播客访谈解说-脚本生成
@File   : script_generation.py
@Description: 基于已规划片段生成播客访谈解说脚本
"""

from ..base import ParameterizedPrompt, PromptMetadata, ModelType, OutputFormat


class ScriptGenerationPrompt(ParameterizedPrompt):
    """播客访谈脚本生成提示词"""

    def __init__(self):
        metadata = PromptMetadata(
            name="script_generation",
            category="podcast_interview",
            version="v1.1",
            description="基于已规划片段生成播客访谈解说脚本，为 OST=0 片段写接/补/点短解说",
            model_type=ModelType.TEXT,
            output_format=OutputFormat.JSON,
            tags=["播客", "访谈", "解说脚本", "接补点", "原声保留"],
            parameters=[
                "drama_name",
                "drama_genre",
                "plot_analysis",
                "subtitle_content",
                "segment_plan",
                "narration_language",
            ],
        )
        super().__init__(
            metadata,
            required_parameters=["drama_name", "plot_analysis", "segment_plan"],
        )

        self._system_prompt = (
            "你是一位播客访谈解说文案写手。你必须严格按照JSON格式输出，"
            "只能补充picture和narration，不能改动上游片段规划中的_id、video_id、video_name、timestamp和OST。"
        )

    def get_template(self) -> str:
        return """# 播客访谈脚本生成任务

## 任务目标
为访谈节目《${drama_name}》生成最终可剪辑解说脚本。片段已经由上游规划完成，
你只能补充 picture 和 narration，不能改变片段来源、时间戳和 OST。

## 输入材料

### 访谈分析材料
<analysis>
${plot_analysis}
</analysis>

### 已规划片段（必须逐项照抄结构字段）
<segment_plan>
${segment_plan}
</segment_plan>

### 原始字幕（访谈逐字稿）
<subtitles>
${subtitle_content}
</subtitles>

### 解说台词语言
<narration_language>
${narration_language}
</narration_language>

### 访谈类型
<drama_genre>
${drama_genre}
</drama_genre>

## 绝对绑定规则
1. 输出 items 数量、顺序和 _id 必须与 segment_plan 完全一致。
2. 每个 item 的 _id、video_id、video_name、timestamp、OST 必须逐字复制 segment_plan，不得新增、合并、拆分或改动。
3. 你只能补充 picture 和 narration 两个字段。
4. OST=1 的 narration 必须写成"播放原片+_id"，例如 _id 为 5 时写"播放原片5"。
5. OST=0 的 narration 必须使用 ${narration_language}，并严格基于访谈分析的字幕事实，不虚构字幕外的具体事件。
6. OST=2 的 narration 需要同时服务解说和原声（过渡段），解说不超过 20 字。

## 红线（违反即废稿）

1. 不得改变嘉宾原话的意思。剪掉限定条件和例外即视为改意。
2. OST=0 的 narration 可以改写，但不得虚构字幕中不存在的具体事件、观点、原话。
3. 反差、矛盾、反问，两个端点都必须是字幕中真实出现过的表达。

## 解说词写作规则（接/补/点）

OST=0 的 narration 只做三件事，其他一律删除：
1. 接：引出紧接在原声前的悬念，告诉观众为什么要听下一段。
2. 补：补充片段中缺失、观众不听就理解不了的背景。
3. 点：点出相邻两段的推进关系。

## 硬约束
1. 每段 OST=0 解说 ≤ 3 句，每句 ≤ 25 字；三句合计 ≤ 60 字。
2. 连接词必须来自 segment_plan 的 relation_to_next 对应组：
   - 限定：但问题在于 / 前提是 / 他自己也承认 / 反过来看
   - 修正：说到这儿他停了一下 / 他紧接着改口 / 后来他修正了
   - 掀翻：但真正有意思的是 / 他根本不认同这个问法
   - 举例：他举了个例子 / 这件事他专门讲了一遍
   - 递进：更进一步 / 更根本的是 / 往深一层说
3. 开头一段 OST=1 之前如果存在一段 OST=0 作为前置钩子（可选），必须简短（≤ 25 字），且不得抢走嘉宾金句的冲击力。
4. 结尾优先停在原声（最后一段是 OST=1），或用一个开放问题收住。
5. 不复述嘉宾原话，不替嘉宾升华，不加人生道理。

## 解说密度与画面节奏
- OST=0 文案必须能被当前 timestamp 的画面承载：
  - 中文：narration 字数 / 5.0 = 所需秒数
  - 英文：narration 词数 / 2.8 = 所需秒数
- 如果画面只有 6 秒，不要写 80 字；压缩到约 30 字（中文）或约 15 词（英文）。
- 长信息要拆成多段，每段只承担一个叙事功能。

## picture 字段
- 描述嘉宾在画面中的动作、表情、场景，便于后期识别素材。
- 不要写镜头调度或剪辑指令。

## 输出格式

请严格按照以下 JSON 格式输出，绝不添加任何其他文字、说明或代码块标记：

{
  "items": [
    {
      "_id": 1,
      "video_id": 1,
      "video_name": "1.mp4",
      "timestamp": "00:05:23,000-00:05:41,000",
      "picture": "嘉宾身体前倾，手指敲击桌面强调观点",
      "narration": "播放原片1",
      "OST": 1
    },
    {
      "_id": 2,
      "video_id": 1,
      "video_name": "1.mp4",
      "timestamp": "00:05:41,000-00:05:48,000",
      "picture": "主持人点头，画面切回嘉宾",
      "narration": "但问题在于，他自己也承认这个判断有前提——",
      "OST": 0
    }
  ]
}

现在请基于以上要求，为访谈节目《${drama_name}》生成解说脚本。"""