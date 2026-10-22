#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: 播客访谈解说-解说正文
@File   : narration_copy.py
@Description: 生成可供用户审核修改的播客访谈解说正文
"""

from ..base import ParameterizedPrompt, PromptMetadata, ModelType, OutputFormat


class NarrationCopyPrompt(ParameterizedPrompt):
    """播客访谈解说正文提示词"""

    def __init__(self):
        metadata = PromptMetadata(
            name="narration_copy",
            category="podcast_interview",
            version="v1.3",
            description="基于访谈分析和字幕生成可审核修改的解说正文（只负责内容质量，不负责结构）",
            model_type=ModelType.TEXT,
            output_format=OutputFormat.TEXT,
            tags=["播客", "访谈", "解说文案", "串场", "接补点", "用户审核"],
            parameters=[
                "drama_name",
                "drama_genre",
                "plot_analysis",
                "subtitle_content",
                "narration_language",
                "narration_word_count",
            ],
        )
        super().__init__(
            metadata,
            required_parameters=["drama_name", "plot_analysis", "subtitle_content"],
        )

        self._system_prompt = (
            "你是一位资深播客内容编辑，擅长用解说串联嘉宾原声。"
            "你的解说只做三件事：接（引出下一段原声）、补（补充背景）、点（点出关系）。"
            "不复述、不总结、不替嘉宾升华。"
            "语言必须口语化、简洁、有节奏感，禁止学术书面语。"
            "只输出纯文本，不要 JSON、时间戳、编号、标题、解释或 Markdown。"
        )

    def get_template(self) -> str:
        return """# 播客访谈解说正文创作任务

## 任务边界（先读这里）

这个阶段**只负责写解说词的内容**，不负责：
- 段落时长、每段词数上限
- OST 分配（原声/解说比例）
- 关系链标注

这些是下游 `script_matching` 阶段的工作。你只需要把**优质的、口语化的、有钩子的**解说词写出来。

## 目标
为访谈节目《${drama_name}》创作一份可直接给用户审核修改的解说正文。

## 访谈分析材料
<analysis>
${plot_analysis}
</analysis>

## 原始字幕（访谈逐字稿）
<subtitles>
${subtitle_content}
</subtitles>

## 输出语言
<narration_language>
${narration_language}
</narration_language>

## 访谈类型
<drama_genre>
${drama_genre}
</drama_genre>

## 用户要求的文案字数
<narration_word_count>
${narration_word_count}
</narration_word_count>

## 红线（违反即废稿）

1. 不得改变嘉宾原话的意思。剪掉限定条件和例外（"这种情况下不建议"→"不建议"）即视为改意。
2. 开头第一句必须是字幕中可逐字定位的嘉宾原话，或对原话的直接转述。不得自行造句。
3. 反差、矛盾、反问，两个端点都必须是字幕中真实出现过的表达。
4. 转写稿中定位不到的原话，不得使用。

## 解说的职责（只干三件事）

| 职责 | 说明 | 反例 |
|---|---|---|
| 接 | 引出下一段原声，告诉观众为什么要听 | ❌ "他接下来会讲到..." |
| 补 | 补充片段中缺失、观众不听就理解不了的背景 | ❌ 复述嘉宾观点 |
| 点 | 点出相邻两段的推进关系 | ❌ "总而言之..." |

**三件事之外的话全部删除。不复述、不总结、不替嘉宾升华。**

## 格式要求（唯一和"结构"相关）

1. **用空行分段**：每段解说独立成段，段与段之间用空行分隔。
2. **每段大致对应一个嘉宾原声片段**：一段解说大约 1-3 句，长度大致匹配一段嘉宾说话的时长（不需要精确，下游会调整）。
3. **不要用编号、项目符号、Markdown 标题**。

**示例格式**：
```
他做过三次创业，两次失败。但他最怕的不是失败——

我当年运气好，换个人不一定成。

第二家店我只拿了很少的钱启动，很多人不理解。

钱少的时候我每一个决定都想得很清楚，钱多的时候我会骗自己。
```

## 内容质量要求

### 1. 口语化，禁用学术书面语

| 禁用词/短语 | 口语化替换 |
|---|---|
| stated flatly | right out of the gate |
| cashes only one | only takes one today |
| the split is clean | it splits cleanly |
| does real work | that distinction matters |
| worth flagging | worth noting |
| the precision is the accusation | that's the point |
| the hour builds on it | he builds the rest on this |

### 2. 减少 "He/She/They + 动词" 开头

**不要每段都用"He says..." / "He thinks..." / "He believes..." 开头。**

**整体控制在 30% 以内**（10 段解说里最多 3 段用这种开头）。

用以下模板替换：
- 悬念："But then he adds something you might not expect—"
- 转折："Here's where it gets weird."
- 追问："So what did he do?"
- 反差："Except it didn't work out that way."
- 时间压力："Then came January 2026."
- 对比："Contrast that with what happened next."

### 3. 句式简短

**单句不超过 20 个英文词 / 30 个中文字符。**

- ❌ "He does not soften the failure. The war went on. But failing to end a war is not owning it; that gap is key."（3 句 60+ 词）
- ✅ "No softening here. The war went on. But failing to end a war is not the same as owning it. That gap matters."（4 句 50 词，句式更短）

### 4. 有推进感

解说词之间要有"接/补/点"的推进感，**用连接词自然过渡**，不需要显式标注"这是限定/递进/举例"。

**推进感示例**：
- "But here's the catch—"（限定）
- "And it goes deeper."（递进）
- "One concrete case proves it."（举例）
- "Except he rejects the question."（掀翻）
- "Then comes the reversal."（转折）

### 5. 总字数控制

**总字数必须在 {narration_word_count} ±15% 范围内。**
- {narration_word_count} 是硬目标
- 超出 15% 必须重写

## 开头四种打法（按优先级）

1. **甩原话**：字幕中存在一句本身即满足钩子的原话 → 直接甩原话，不加铺垫
2. **反差**：存在一对真实矛盾事实 → "一个……的人，为什么……"
3. **掀前提**：嘉宾否定了提问本身 → "主持人问他……，他说：这个问题本身就有问题"
4. **原声切入**：以上均无，直接切入最有张力的原声瞬间

**禁止开头**："今天我们来聊一聊"、"这期节目请到了"。

## 结尾四种收法（按优先级）

1. **停在原声**（优先）：最后一段原声播完直接结束，不加解说
2. **一句解说收住**：仅当最后一段原声超过 25 字 / 15 词时启用
3. **留一个开放问题**：访谈本身无结论时使用
4. **停在分歧上**：嘉宾与主持人未吵完时使用

**结尾三不要**：
- 不强行加人生道理
- 不总结"这期视频讲了三点"
- 不写"值得一看"

## 访谈类型写作重点（按用户选择的方向，不要自行改判）
- 观点/思想：突出论点推进、概念澄清、反直觉判断
- 故事/经历：突出关键事件、转折点、亲历细节
- 商业/创业：突出决策时刻、失败教训、反常识经验
- 科技/学术：突出核心洞察、机制解释、边界条件
- 文化/生活：突出个人选择、社会观察、情绪共鸣
- 自定义类型：严格服从用户填写的方向

## 输出要求

**只输出解说正文，段落之间用空行分隔。**

不要输出 JSON、时间戳、代码块、编号或任何解释。

## 自检清单（输出前逐条核验，任一不通过即重写）

- [ ] 开头第一句可在字幕中逐字定位？
- [ ] 总字数在目标 ±15% 内？
- [ ] 单句不超过 20 词（英文）/ 30 字（中文）？
- [ ] "He/She/They" 开头不超过 30%？
- [ ] 没有学术书面语禁用词？
- [ ] 用空行分段，每段大致对应一段原声？"""
