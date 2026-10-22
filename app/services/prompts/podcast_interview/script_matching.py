#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: 播客访谈解说-文案画面匹配
@File   : script_matching.py
@Description: 将用户审核后的访谈解说文案匹配到字幕时间戳并生成最终剪辑脚本
"""

from ..base import ParameterizedPrompt, PromptMetadata, ModelType, OutputFormat


class ScriptMatchingPrompt(ParameterizedPrompt):
    """播客访谈解说文案画面匹配提示词"""

    def __init__(self):
        metadata = PromptMetadata(
            name="script_matching",
            category="podcast_interview",
            version="v1.3",
            description="把审核后的播客访谈解说文案匹配到原声片段时间轴，输出以原声为主体的剪辑JSON，强制节奏变化与关系链多样性",
            model_type=ModelType.TEXT,
            output_format=OutputFormat.JSON,
            tags=["播客", "访谈", "画面匹配", "剪辑脚本", "时间戳", "原声为主", "节奏"],
            parameters=[
                "drama_name",
                "drama_genre",
                "plot_analysis",
                "subtitle_content",
                "narration_copy",
                "narration_language",
                "original_sound_ratio",
            ],
        )
        super().__init__(
            metadata,
            required_parameters=["drama_name", "subtitle_content", "narration_copy"],
        )

        self._system_prompt = (
            "你是一位懂播客访谈叙事节奏的剪辑师。你必须严格输出JSON。"
            "访谈的原声是核心价值，必须大量保留；解说只负责串场、补充背景和点出关系。"
            "你必须在节奏和结构上做出变化，禁止机械 1:1 交替。"
        )

    def get_template(self) -> str:
        return """# 播客访谈解说文案画面匹配任务

## 目标
用户已经审核并修改了解说文案。请根据这份文案和原始字幕，生成最终可剪辑 JSON 脚本。

## 节目名
${drama_name}

## 访谈分析材料
<analysis>
${plot_analysis}
</analysis>

## 用户审核后的解说文案
<narration_copy>
${narration_copy}
</narration_copy>

## 原始字幕（访谈逐字稿，含视频编号和局部时间戳）
<subtitles>
${subtitle_content}
</subtitles>

## 输出语言
<narration_language>
${narration_language}
</narration_language>

## 用户选择的访谈类型
<drama_genre>
${drama_genre}
</drama_genre>

## 用户选择的原声占比
<original_sound_ratio>
${original_sound_ratio}%
</original_sound_ratio>

## OST 分配策略（访谈专用）

| OST | 含义 | 目标占比 | 用途 |
|---|---|---|---|
| 1 | 纯原声 | 55-65% | 嘉宾说话是核心价值 |
| 0 | 纯解说 | 25-35% | 主持人串场、观点提炼 |
| 2 | 解说+原声 | 5-10% | 解说需要原声背景的过渡段 |

即使配置值不同，访谈类内容也必须遵守：
- OST=1 的总时长占全片 ≥ 50%
- OST=0 的总时长占全片 ≤ 40%

## 节奏约束（建议满足，不强制）

访谈的真实剪辑节奏是变化的。请在以下几种节奏模式中至少使用 2 种，避免全片机械 1:1 交替：

- **2 段连续原声**：嘉宾连续说完一个完整意思，主持人克制不打断
- **3 段连续原声**：嘉宾长段独白
- **2 段连续解说**：主持人需要补充大段背景
- **1:1 交替**：对话式场景使用

**示例节奏分布**（合法）：
```
items 排列: 1  1  0  1  1  1  0  0  1  1  0  1  0  0  1  1
             ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
             连续段:  2连原声  1  3连原声  2连解说  2连原声  1  1  2连解说  2连原声
```

如果输出是全片 `1,0,1,0,1,0,...` 的机械交替，请重新调整。

## 解说词长度硬约束

对每个 OST=0 片段，请按以下规则控制解说词长度：

| 语言 | 每分钟正常解说词数 | 每片段最大解说词数 |
|---|---|---|
| 英文 | 约 150 词 | 片段时长的秒数 × 2.5 |
| 中文 | 约 300 字 | 片段时长的秒数 × 5.0 |

**举例**：
- 英文：片段 6 秒 → 解说词最多 15 个英文单词
- 中文：片段 6 秒 → 解说词最多 30 个中文字符（不含空格）

**超过上限时，优先从字幕里找一个更长的窗口，让片段时间戳覆盖更长的原声；次选缩短解说词。**

## 关系链多样性

相邻两段之间必须属于以下五种关系之一，全片至少覆盖 3 种：

| 关系 | 判定问题 | 典型表现 |
|---|---|---|
| 限定 | 后段是否改变了前段的结论？ | 嘉宾先说 A，后说"但前提是..." |
| 递进 | 后段是否把论述推向更根本的原因？ | 嘉宾先说现象，后说根本原因 |
| 举例 | 后段是否是前段的具体例子？ | 嘉宾先说原则，后举具体案例 |
| 掀翻 | 后段是否否定了前段的前提？ | 嘉宾先被问 X，后说"这个问题本身有问题" |
| 转折 | 后段是否是反向/相反的情况？ | 嘉宾先说成功案例，后说失败案例 |

**禁止全片只用"并列"关系。**

## 开头必须是最强原声钩子

**第一个片段必须是 OST=1，且必须是全片最有张力的原声片段。**

优先级：
1. **绝对判断**（"America loses all three wars."）
2. **反直觉判断**（"The best thing was the war ending, not starting."）
3. **情绪爆发**（嘉宾提高音量/明显情绪）
4. **矛盾对撞**（"He says X, but two minutes earlier said not-X"）

**不要用"总判断/概述"开场**——那是中间段的内容，不是开场。

## 结尾规则

- **优先**：最后一段 OST=1 播完直接结束，不加解说
- **次选**：最后一段 OST=0 是一句不超过 10 个英文单词 / 15 个中文字符的开放问题
- **禁止**：长段解说收尾，或"总结三点"式收尾

## 匹配流程

1. 先按句号、问号、感叹号、省略号切分解说文案，得到候选解说句。
2. 逗号只在明显分割两个动作、场景、观点或描述对象时切分。
3. 不要求每个候选句都单独输出为 OST=0；可以合并、压缩相邻候选句。
4. 严禁把解说文案匹配到片头、片尾、演职员表、版权声明、平台水印展示、下集预告、花絮、赞助口播、商品露出、贴片广告、中插广告、片中广告或任何与主线无关的推广片段。
5. 如果字幕出现"广告""赞助""推广""片头""片尾""预告""下集""扫码""购买""会员""关注"等非主线信号，跳过对应时间段。
6. 为每个解说片段寻找最匹配的原始字幕画面。
7. timestamp 必须使用对应 video_id 内部局部时间戳。
8. 同一 video_id 内时间段不得交叉或重叠。
9. 不要自行判断或改写访谈类型。

## 字段规则
- _id：从 1 开始连续递增。
- video_id：来自字幕分段标题。
- video_name：对应视频文件名。
- timestamp：格式为 "HH:MM:SS,mmm-HH:MM:SS,mmm"。
- picture：描述画面中嘉宾、主持人、动作、情绪、场景和关键道具。
- narration：OST=0 时填写用户文案片段；OST=1 时填写"播放原片+_id"；OST=2 时填写不超过 20 字的过渡解说。
- OST：解说片段填 0，原声片段填 1，过渡段填 2。

## 输出格式
只输出严格 JSON：

{
  "items": [
    {
      "_id": 1,
      "video_id": 1,
      "video_name": "1.mp4",
      "timestamp": "00:08:29,840-00:08:41,240",
      "picture": "嘉宾身体前倾，手指敲击桌面强调观点",
      "narration": "播放原片1",
      "OST": 1
    },
    {
      "_id": 2,
      "video_id": 1,
      "video_name": "1.mp4",
      "timestamp": "00:08:41,240-00:08:46,240",
      "picture": "主持人点头，画面切回嘉宾",
      "narration": "But then he adds something you might not expect—",
      "OST": 0
    }
  ]
}

## 输出前自检（能通过 6 项以上即可，不要求全过）

- [ ] 节奏：有连续 2 段或 3 段原声的场景？
- [ ] 节奏：有连续 2 段解说的场景？
- [ ] 长度：OST=0 解说词没超过"片段时长 × 2.5 词（英文）/ × 5.0 字（中文）"？
- [ ] 关系链：至少覆盖 3 种（限定/递进/举例/掀翻/转折）？
- [ ] 开头：第一段是 OST=1 且是全片最有张力的原声？
- [ ] 结尾：最后一段是 OST=1 或一句很短的开放问题？
- [ ] OST=1 总时长 ≥ 50%，OST=0 总时长 ≤ 40%？
- [ ] 每个 OST=1 的 narration 都是"播放原片{_id}"格式？
- [ ] 相邻片段时间戳不重叠？

现在请基于用户审核后的解说文案生成最终剪辑脚本。"""
