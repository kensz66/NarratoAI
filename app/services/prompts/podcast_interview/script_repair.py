#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: 播客访谈解说-脚本修复
@File   : script_repair.py
@Description: 播客访谈脚本校验失败后的JSON修复提示词，含结构性死板诊断
"""

from ..base import ParameterizedPrompt, PromptMetadata, ModelType, OutputFormat


class ScriptRepairPrompt(ParameterizedPrompt):
    """播客访谈脚本修复提示词"""

    def __init__(self):
        metadata = PromptMetadata(
            name="script_repair",
            category="podcast_interview",
            version="v1.3",
            description="根据确定性校验错误修复播客访谈脚本JSON，含结构性死板诊断与节奏强制改变",
            model_type=ModelType.TEXT,
            output_format=OutputFormat.JSON,
            tags=["播客", "访谈", "脚本修复", "JSON", "原声占比", "解说密度", "节奏诊断"],
            parameters=[
                "drama_name",
                "drama_genre",
                "plot_analysis",
                "subtitle_content",
                "invalid_script",
                "validation_errors",
                "narration_language",
            ],
        )
        super().__init__(
            metadata,
            required_parameters=[
                "drama_name",
                "subtitle_content",
                "invalid_script",
                "validation_errors",
            ],
        )

        self._system_prompt = (
            "你是一位播客访谈脚本JSON修复器。你只能根据校验错误修复JSON，"
            "必须输出严格JSON，不能输出解释、Markdown或代码块。"
            "修复时优先诊断并打破结构性死板（1:1 交替、全复述、关系链单一）。"
        )

    def get_template(self) -> str:
        return """# 播客访谈脚本修复任务

## 修复目标
下面的访谈节目《${drama_name}》解说脚本未通过剪辑校验。请只根据校验错误和字幕内容修复它，
输出一个完整可剪辑的 JSON。

## 访谈分析材料
<analysis>
${plot_analysis}
</analysis>

## 校验错误
<validation_errors>
${validation_errors}
</validation_errors>

## 当前无效脚本
<invalid_script>
${invalid_script}
</invalid_script>

## 可用字幕窗口
<subtitles>
${subtitle_content}
</subtitles>

## 解说台词目标语言
<narration_language>
${narration_language}
</narration_language>

## 用户选择的访谈类型
<drama_genre>
${drama_genre}
</drama_genre>

## 修复前先诊断（4 项，用于决定修复方向）

### 诊断 1：解说词是否超密
- 逐条检查所有 OST=0 片段
- 对每个片段，计算"解说词的字数（英文按单词数，中文按去掉空格的字符数）"
- 与"片段时长的秒数"对比
- 如果解说词字数 > 片段时长秒数 × 2.5（英文）/ × 5.0（中文）→ 该片段超密

**修复方式（二选一）**：
- A. **缩短解说词**：保持原意，删掉修饰词、重复句
- B. **延长片段时间戳**：从字幕里找更长的窗口（比如当前片段是 `00:01:00-00:01:06`，字幕里这段后面还有 5 秒的话没被选中，就把结束时间延到 `00:01:11`）

### 诊断 2：解说词是否全复述
- 逐条检查所有 OST=0 片段的解说词开头
- 统计以"He said / She said / He thinks / She believes / He is / He was / He does"等形式开头的数量

**修复方式**：把这些开头改写成口语化模板（见下方"口语化重写对照表"）

### 诊断 3：节奏是否机械 1:1 交替
- 遍历所有 items，统计相邻两段的 OST 是否不同
- 如果**几乎每一对相邻片段**都是 OST 不同（比如 90%+ 的相邻对都是 `0→1` 或 `1→0`）→ 机械交替

**修复方式（选一种）**：
- A. **合并相邻原声**：找到两个相邻的 OST=1（比如 _id=3 和 _id=5），把它们中间的 OST=0（_id=4）删掉或挪走，让 _id=3 和 _id=5 直接相连
- B. **拆分连续解说**：把一段长 OST=0 拆成两段，中间插入一个 OST=1

### 诊断 4：关系链是否单一
- 逐条检查所有 OST=0 的解说词，判断它和上一段、下一段的关系属于：限定 / 递进 / 举例 / 掀翻 / 转折 / 并列
- 如果超过一半是"并列"关系 → 关系链单一

**修复方式**：给部分解说词的开头加关系词（见下方"关系词改写对照表"）

## 修复规则（按优先级）

### 通用规则
1. 只输出 JSON，不要任何解释、标题、Markdown 或代码块。
2. 输出根对象必须是 `{"items": [...]}`。
3. 每个 item 必须包含 _id、video_id、video_name、timestamp、picture、narration、OST。
4. 必须删除片头、片尾、演职员表、版权声明、平台水印展示、下集预告、花絮、赞助口播、商品露出、贴片广告、中插广告、片中广告或任何与主线无关的推广片段。
5. 如果字幕出现"广告""赞助""推广""片头""片尾""预告""下集""扫码""购买""会员""关注"等非主线信号，必须删除对应 item。
6. video_id、video_name 和 timestamp 必须来自对应字幕窗口；不得把不同视频的同名时间戳混用。
7. 同一 video_id 内片段不得交叉或重叠。
8. 尽量保留原脚本中没有错误的片段；无法修复的片段可以删除，但剩余片段必须重新按 1 开始编号。

### 访谈专用规则
9. **第一个片段必须是 OST=1 嘉宾原声钩子**，且必须是最有张力的原声（绝对判断/反直觉/情绪爆发）。
10. OST=1 的 narration 必须是"播放原片+序号"；OST=0 的 narration 必须使用 ${narration_language}。
11. **访谈类 OST 分布下限：OST=1 总时长 ≥ 50%，OST=0 总时长 ≤ 40%。**
12. **允许连续 2 段甚至 3 段 OST=1**（嘉宾连续说话时，不要硬插解说）。
13. **允许连续 2 段 OST=0**（主持人补充大段背景时）。
14. **跨 video_id 切换前后不能都是 OST=1**；必须至少有一个 OST=0 片段解释场景切换。
15. **解说密度修复**：按上面"诊断 1"的两种方式处理。
16. **原声不足修复**：在时间窗内寻找能支撑相邻解说的嘉宾原话，插入为 OST=1 片段；不得编造字幕中不存在的原话。

## 口语化重写对照表（诊断 2 超标时使用）

| 原写法 | 口语化替换 |
|---|---|
| He says X | Here's what he says— / Then comes this— |
| He is careful about Y | Watch how careful he is about Y |
| He reaches for Z | Then Z shows up / Cue Z |
| He splits A into B | A splits into B |
| He does not soften the failure | No softening here. |
| He is explicit that... | He makes it clear— |
| He now runs the same... | Same four questions, three wars. |
| The split is clean | It splits cleanly |
| That gap is key | That's the whole story |
| Worth flagging | Worth noting |
| The precision is the accusation | That's the point |
| That care does real work | That distinction matters |

## 关系词改写对照表（诊断 4 超标时使用）

| 当前关系 | 改写方向 | 开头示例 |
|---|---|---|
| 并列 | → 限定 | But the premise has a catch. / Except there's a condition. |
| 并列 | → 递进 | And it goes deeper. / The real reason is further back. |
| 并列 | → 举例 | One concrete case proves it. / Here's the example he gives. |
| 并列 | → 掀翻 | Except he rejects the question. / He says the question itself is wrong. |
| 并列 | → 转折 | Then comes the reversal. / But reality went the other way. |

## 修复步骤（按顺序执行）

1. **先修诊断 1（解说超密）**：逐条缩短超密的解说词，或延长片段时间戳
2. **再修诊断 3（机械交替）**：合并相邻 OST=1 或拆分 OST=0，打破 1:1
3. **再修诊断 2（全复述）**：改写以"He/She/They"开头的解说词
4. **最后修诊断 4（关系链单一）**：给部分解说词加关系词
5. **输出前**：按下面"交稿前自查清单"逐项核验

## 交稿前自查清单

- [ ] 第一个片段是 OST=1 且是全片最有张力的原声？
- [ ] 至少有一处 2 段或 3 段连续原声？
- [ ] 至少有一处 2 段连续解说？
- [ ] OST=0 的解说词都没超过"片段时长 × 2.5 词（英文）/ × 5.0 字（中文）"？
- [ ] "He/She/They" 开头的解说词少于 20%？
- [ ] 关系链至少覆盖 3 种？
- [ ] 结尾停在原声或一句很短的开放问题上？
- [ ] OST=1 总时长 ≥ 50%，OST=0 总时长 ≤ 40%？

请输出修复后的完整 JSON。"""
