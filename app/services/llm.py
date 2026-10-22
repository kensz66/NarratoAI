import os
import re
import json
import traceback
import streamlit as st
from typing import List
from loguru import logger
from openai import OpenAI
from openai import AzureOpenAI
from moviepy import VideoFileClip
from openai.types.chat import ChatCompletion
import google.generativeai as gemini
from googleapiclient.errors import ResumableUploadError
from google.api_core.exceptions import *
from google.generativeai.types import *
import subprocess
from typing import Union, TextIO

from app.config import config
from app.utils.openai_base_url_security import (
    openai_compatible_base_url_warning,
    validate_openai_compatible_base_url,
)
from app.utils.utils import clean_model_output

_max_retries = 5

Method = """
# 访谈播客解说文案写作全流程（GPT 执行版）

## 零、任务定义与红线

你要做的是：从访谈转写稿里挑出若干片段，重新排序，用解说词把它们连成一条有推进关系的观点线。不是复述对话。

红线（违反即废稿）：
1. 不得改变嘉宾原话的意思。剪掉限定条件和例外（"这种情况下不建议"→"不建议"、"我当年运气好"→"我成功了"）即视为改意。
2. 开头第一句必须是转写稿中可逐字定位的嘉宾原话，或对原话的直接转述。不得自行造句、不得把语境加工成新句子。
3. 反差、矛盾、反问，两个端点都必须是转写稿中真实出现过的表达。

---

## 一、定主线

先回答：这条片子讲清楚哪一件事？用一句话写出主线。

### 判定标准（全过才算定住）

1. 一句话说完，不含"而且""另外""同时"。
2. 含嘉宾的判断动词（认为/主张/反对/建议/否定），不是话题。
   - 反例："他聊了创业经历"（话题）
   - 正例："他认为创业者最大的敌人是过早拿到钱"（判断）
3. 能对应到转写稿中 2 段以上具体片段，能指出是哪几段。
4. 可反问测试：把主线写成一句陈述，检查它是否必然引出一个"为什么/凭什么"的追问。引不出追问，主线不合格。

### 故事型主线的例外

若嘉宾讲一段完整经历，主线可写成因果链（"因为 X，所以他 Y"）。对应片段放宽为：1 段长叙述 + 1 段能独立复述、且直接支持该因果链的佐证片段。

### 检查点

- 主线不含判断动词 → 回第一条重写。
- 主线写不出它该引出的追问 → 回第一条重写。
- 主线对应不到 2 段以上片段 → 回第一条重写。

---

## 二、选片段

### 标准

片段单独成立测试：把该段从上下文中拿出，检查它是否包含一个可独立复述的完整意思。
- 可独立复述、意思不变 → 留。
- 不可独立复述，或脱离上下文变味 → 用一句解说补背景后仍不可独立成立，则弃用。

### 数量

不预设段数。段数由第三节的关系链推导得出，并受第七节时长容量约束。本节只负责判断"哪些片段有资格进链"，不决定进几段。

### 检查点

每段必须能通过"片段单独成立测试"，否则不得进入排序。

---

## 三、排顺序与定段数（核心步骤）

### 3.1 先标关系

相邻两段之间必须且只能属于以下五种关系之一。先判定关系，再写解说词。

| 关系 | 判定问题 | 连接词 |
|---|---|---|
| 限定 | 后段是否改变了前段的结论（前段对但不全对）？ | 但问题在于 / 前提是 / 他自己也承认 / 这话有个条件 / 反过来看 / 值得注意的是 |
| 修正 | 后段是否改口、收回或修正了前段的说法？ | 说到这儿他停了一下 / 他紧接着改口 / 后来他修正了这个说法 / 其实也不全是这样 |
| 掀翻 | 后段是否否定了前段的前提或提问本身？ | 但真正有意思的是 / 他根本不认同这个问法 / 他说这个问题本身就不成立 / 聊到一半他打断了主持人 |
| 举例 | 后段是否是前段的具体例子或证据？ | 他举了个例子 / 这件事他专门讲了一遍 / 他自己就是这么做的 / 后来发生的一件事印证了这句话 |
| 递进 | 后段是否未改变前段结论、但把论述推向更根本的原因？ | 更进一步 / 更根本的是 / 往深一层说 / 说到根上 |

限定与递进的排他判据（必须二选一，不得并存）：
- 先问：后段有没有改变前段的结论？
  - 改变了 → 限定。
  - 没改变，只是把前段推向更根本的原因 → 递进。
- 两者都判不出 → 降级为并列，按 3.3 处理。

连接词只作提示，最终以关系判定为准。连接词与判定冲突时，以判定为准。

### 3.2 由关系链推导段数

1. 写出主线需要回答的那个追问（第一节）。
2. 列出回答该追问所需的最少推进次数。每需要一次推进，增加一段。
3. 把关系链写成一行：A —关系→ B —关系→ C。这一行有几个节点，段数就是几。
4. 用第七节算出的时长容量校验：段数超出容量 → 砍掉关系链中最弱的一环；砍到撑不住主线 → 触发第八节回退，收窄主线。

段数是关系链的输出，不是输入。访谈类型不改变段数，只改变段内排法：观点型每段短、推进靠限定/掀翻；故事型每段长、推进靠转折/举例。

### 3.3 硬规则

- 相邻两段标不出五种关系中的任何一种 → 该段不得进链，段数相应减少。
- 并列最多出现一次（最多两段并排）。并列第二次出现 → 关系链已断，回 3.2 重推。
- 关系链减到撑不住主线 → 触发第八节回退。

### 检查点

把关系链写成 A —关系→ B —关系→ C 一行字。写不出这一行，段数未定住，不得进入第四节。

---

## 四、写解说词

### 只干三件事

1. 接：引出下一段，告诉观众为什么接下来要听这一段。
2. 补：补充片段中缺失、观众不听就理解不了的背景。
3. 点：点出相邻两段的推进关系，使用第三节对应关系组的连接词。

三件事之外的话全部删除。不复述、不总结、不替嘉宾升华。

### 硬约束

- 接、补、点每句不超过 25 字；三句合计不超过 60 字。
- 连接词必须先判定关系、再从对应组选取，不得跨组混用。

### 原声剪辑

保留原话，剪掉语气词、重复、离题部分。限定条件和例外必须保留。
- 正例："我当年运气好，换个人不一定成" → 可剪为"我当年是运气好"。
- 反例：剪为"我成功了"。

### 检查点

遮住原声，只读解说词。若解说词自身构成一个完整故事，说明写多了，删至只剩接、补、点。

---

## 五、开头

### 判定标准

第一句必须满足下列结构特征之一，否则重写：
1. 含一个反直觉判断（X 其实是 Y）。
2. 含一对矛盾事实（A 却做了非 A 的事）。
3. 是一句完整的、带转折的嘉宾引语。

禁止开头："今天我们来聊一聊""这期节目请到了某某"。

### 四种打法，按顺序匹配，第一个命中的即用

1. 转写稿中存在一句本身即满足上述结构特征的原话 → 直接甩原话，不加铺垫。
2. 无此类原话，但存在一对真实矛盾事实 → 用反差句式（"一个……的人，为什么……"）。
3. 嘉宾否定了提问本身 → 用掀前提句式（"主持人问他……，他说：这个问题本身就有问题"）。
4. 以上均无，但存在一个画面感强的原声瞬间 → 直接从该原声切入。

### 红线

- 第一句必须是转写稿中可逐字定位的原话或直接转述。
- 反差两端必须是转写稿中真实出现过的表达。
- 转写稿中定位不到的原话，不得使用。

### 检查点

第一句对照上述三种结构特征逐条核验，不满足任一 → 重写。第一句无法在转写稿中逐字定位 → 重写。

---

## 六、结尾

### 任务

让观众带走一句话。禁止"以上就是本期内容"。

### 四种收法，按优先级选用

1. 停在原声（优先）。最后一段原声播完直接结束，不加解说。
2. 一句解说收住。仅当最后一段原声超过 25 字、或含两个以上分句时启用（长句停不住）。
3. 留一个开放问题。访谈本身无结论时，停在嘉宾未答完的问题或与主持人的分歧处。
4. 停在分歧上。嘉宾与主持人未吵完，原样端出分歧，不替任何一方下结论。

### 三不要

- 不强行加人生道理（嘉宾自己说的哲理可用，未说的不得替他升华）。
- 不总结"这期视频讲了三点"。
- 不写"值得一看"。

### 检查点

结尾是否停在上述四种之一？是否出现"以上就是"？出现即删。

---

## 七、时长

### 比例

成片压到原片的 1/4—1/3。

### 三步

1. 算区间：原片 40 分钟 → 10—13 分钟；90 分钟 → 22—30 分钟。得到的是区间。
2. 按类型选位：观点型往 1/4 靠；故事型往 1/3 靠。
3. 按平台微调：短视频信息流再压一档；中视频取区间中段。未提供平台信息时，取区间中段。

### 容量换算（供第三节 3.2 校验使用）

每 3—4 分钟成片容得下 1 个原声片段及其间的解说。按此换算，得到本片段的段数上限。

### 约束

- 新手红线：算出来超过 8 分钟的，不接。
- 第一版取区间下限。宁可短而完整，不可长而散。

---

## 八、质检与回退

### 盲测

用一句"这个人到底说了什么"检验成片，所得结论必须与第一节主线一致。

无法外部验证时，执行静态自检：从成片中提取主讲人的判断句，检查其是否与主线表述一致。不一致 → 按下列顺序排查。

### 回退排查顺序

1. 主线能否被一句话复述、且含判断动词、且能引出追问？不能 → 回第一节。
2. 每个片段能否独立成立？不能 → 回第二节。
3. 每一对相邻片段的关系能否一句话写出？不能 → 回第三节。
4. 段数是否超出第七节容量？超出 → 回第三节 3.2，砍最弱一环或收窄主线。

### 交稿前自查清单

- [ ] 主线含判断动词，且能引出一个追问。
- [ ] 关系链能写成 A —关系→ B —关系→ C 一行字。
- [ ] 每一对相邻片段的关系已标出且互不冲突（限定/递进已二选一）。
- [ ] 遮住原声，解说词自身不构成完整故事。
- [ ] 开头第一句满足三种结构特征之一，且可在转写稿中逐字定位。
- [ ] 结尾停在一句话上，无"以上就是"。

---

## 附：完整样例

素材：一位做了二十年餐饮的老板的访谈转写稿，选出三段：
- 片段 A：关掉第一家店时，反而松了一口气。
- 片段 B：开第二家店时，故意把启动资金压到最低。
- 片段 C：有人问他怎么开店，第一句都是"先想清楚你为什么要开"。

主线：他认为开店最大的风险不是没钱，是太早有钱。（含判断动词，能引出"钱多怎么会是风险"的追问，对应 A、B、C 三段）

关系链：A —限定→ B —举例/递进→ C（A→B：后段改变了前段结论，"松口气"不是因为不信生意，是不信自己配得上那笔钱 → 限定；B→C：C 既是 B 的印证，又推向更根本的原因 → 举例兼递进，按排他判据取递进，举例由解说补出）。

段数：关系链 3 个节点 → 3 段。对照时长容量校验通过。

文案：

【开头·打法 1：甩原话】
他关掉第一家店的那天，反而松了一口气。这家店他做了四年，赔进去的钱够在二线城市买套房。

【解说·接】
一个做了二十年餐饮的人，为什么把"关店"说成解脱？他后来在节目里讲了一件更反常识的事。

【原声 A】
"关掉那天我走出门，第一反应是，终于不用再撑了。那四年我不是在开店，我是在撑一个我不相信的东西。"

【解说·点（限定）】
但问题在于，他说的"不相信"，不是不相信这门生意，是不相信自己当时的能力配得上那笔钱。

【原声 B】
"第二家店我只拿了很少的钱启动，很多人不理解，说你这不是跟自己过不去吗。我说不是，钱少的时候我每一个决定都想得很清楚，钱多的时候我会骗自己。"

【解说·点（举例）】
他举了个例子：第一家店装修超预算，他当时的想法是"反正还能撑"；第二家店超预算，他停了半个月重新算账。

【原声 C】
"现在有人问我怎么开店，我第一句都是，先想清楚你为什么要开。这个问题答不好，钱越多死得越快。"

【结尾·停原声】
直接停在"钱越多死得越快"上，结束。

### 短样例：反差开头（打法 2）

【开头·打法 2：反差】
一个做了三十年教育的人，为什么在节目里说，他不建议自己的孩子上大学？

【解说·接】
他给出的理由，和"读书无用"没有半点关系。

【原声】
"我不是说大学没用，我是说，一个人如果不知道自己为什么要上，那四年只是把这个问题往后推了四年。"

【结尾·停原声】
停在"把这个问题往后推了四年"上，结束。

---

## 最后一条

成片导出前，用"这个人到底说了什么"做一次检验。所得结论与第一节主线一致，这条才算成立。无法外部验证时，执行第八节静态自检。
"""

def handle_exception(err):
    if isinstance(err, PermissionDenied):
        raise Exception("403 用户没有权限访问该资源")
    elif isinstance(err, ResourceExhausted):
        raise Exception("429 您的配额已用尽。请稍后重试。请考虑设置自动重试来处理这些错误")
    elif isinstance(err, InvalidArgument):
        raise Exception("400 参数无效。例如，文件过大，超出了载荷大小限制。另一个事件提供了无效的 API 密钥。")
    elif isinstance(err, AlreadyExists):
        raise Exception("409 已存在具有相同 ID 的已调参模型。对新模型进行调参时，请指定唯一的模型 ID。")
    elif isinstance(err, RetryError):
        raise Exception("使用不支持 gRPC 的代理时可能会引起此错误。请尝试将 REST 传输与 genai.configure(..., transport=rest) 搭配使用。")
    elif isinstance(err, BlockedPromptException):
        raise Exception("400 出于安全原因，该提示已被屏蔽。")
    elif isinstance(err, BrokenResponseError):
        raise Exception("500 流式传输响应已损坏。在访问需要完整响应的内容（例如聊天记录）时引发。查看堆栈轨迹中提供的错误详情。")
    elif isinstance(err, IncompleteIterationError):
        raise Exception("500 访问需要完整 API 响应但流式响应尚未完全迭代的内容时引发。对响应对象调用 resolve() 以使用迭代器。")
    elif isinstance(err, ConnectionError):
        raise Exception("网络连接错误, 请检查您的网络连接(建议使用 NarratoAI 官方提供的 url)")
    else:
        raise Exception(f"大模型请求失败, 下面是具体报错信息: \n\n{traceback.format_exc()}")


def _generate_response(prompt: str, llm_provider: str = None) -> str:
    """
    调用大模型通用方法
        prompt：
        llm_provider：
    """
    content = ""
    if not llm_provider:
        llm_provider = config.app.get("llm_provider", "openai")
    logger.info(f"llm provider: {llm_provider}")
    if llm_provider == "g4f":
        model_name = config.app.get("g4f_model_name", "")
        if not model_name:
            model_name = "gpt-3.5-turbo-16k-0613"
        import g4f

        content = g4f.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
    else:
        api_version = ""  # for azure
        if llm_provider == "moonshot":
            api_key = config.app.get("moonshot_api_key")
            model_name = config.app.get("moonshot_model_name")
            base_url = "https://api.moonshot.cn/v1"
        elif llm_provider == "ollama":
            # api_key = config.app.get("openai_api_key")
            api_key = "ollama"  # any string works but you are required to have one
            model_name = config.app.get("ollama_model_name")
            base_url = config.app.get("ollama_base_url", "")
            if not base_url:
                base_url = "http://localhost:11434/v1"
        elif llm_provider == "openai":
            api_key = config.app.get("openai_api_key")
            model_name = config.app.get("openai_model_name")
            base_url = config.app.get("openai_base_url", "")
            if not base_url:
                base_url = "https://api.openai.com/v1"
        elif llm_provider == "oneapi":
            api_key = config.app.get("oneapi_api_key")
            model_name = config.app.get("oneapi_model_name")
            base_url = config.app.get("oneapi_base_url", "")
        elif llm_provider == "azure":
            api_key = config.app.get("azure_api_key")
            model_name = config.app.get("azure_model_name")
            base_url = config.app.get("azure_base_url", "")
            api_version = config.app.get("azure_api_version", "2024-02-15-preview")
        elif llm_provider == "gemini":
            api_key = config.app.get("gemini_api_key")
            model_name = config.app.get("gemini_model_name")
            base_url = "***"
        elif llm_provider == "qwen":
            api_key = config.app.get("qwen_api_key")
            model_name = config.app.get("qwen_model_name")
            base_url = "***"
        elif llm_provider == "cloudflare":
            api_key = config.app.get("cloudflare_api_key")
            model_name = config.app.get("cloudflare_model_name")
            account_id = config.app.get("cloudflare_account_id")
            base_url = "***"
        elif llm_provider == "deepseek":
            api_key = config.app.get("deepseek_api_key")
            model_name = config.app.get("deepseek_model_name")
            base_url = config.app.get("deepseek_base_url")
            if not base_url:
                base_url = "https://api.deepseek.com"
        elif llm_provider == "ernie":
            api_key = config.app.get("ernie_api_key")
            secret_key = config.app.get("ernie_secret_key")
            base_url = config.app.get("ernie_base_url")
            model_name = "***"
            if not secret_key:
                raise ValueError(
                    f"{llm_provider}: secret_key is not set, please set it in the config.toml file."
                )
        else:
            raise ValueError(
                "llm_provider is not set, please set it in the config.toml file."
            )

        if not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        if llm_provider == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            dashscope.api_key = api_key
            response = dashscope.Generation.call(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    content = response["output"]["text"]
                    return content.replace("\n", "")
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if llm_provider == "gemini":
            import google.generativeai as genai

            genai.configure(api_key=api_key, transport="rest")

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=safety_settings,
            )

            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as err:
                return handle_exception(err)

        if llm_provider == "cloudflare":
            import requests

            response = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "messages": [
                        {"role": "system", "content": "You are a friendly assistant"},
                        {"role": "user", "content": prompt},
                    ]
                },
            )
            result = response.json()
            logger.info(result)
            return result["result"]["response"]

        if llm_provider == "ernie":
            import requests

            params = {
                "grant_type": "client_credentials",
                "client_id": api_key,
                "client_secret": secret_key,
            }
            access_token = (
                requests.post("https://aip.baidubce.com/oauth/2.0/token", params=params)
                .json()
                .get("access_token")
            )
            url = f"{base_url}?access_token={access_token}"

            payload = json.dumps(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                    "top_p": 0.8,
                    "penalty_score": 1,
                    "disable_search": False,
                    "enable_citation": False,
                    "response_format": "text",
                }
            )
            headers = {"Content-Type": "application/json"}

            response = requests.request(
                "POST", url, headers=headers, data=payload
            ).json()
            return response.get("result")

        if llm_provider == "azure":
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
            )
        else:
            base_url = validate_openai_compatible_base_url(base_url)
            warning = openai_compatible_base_url_warning(base_url)
            if warning:
                logger.warning(warning)
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                content = response.choices[0].message.content
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    return content.replace("\n", "")


def _generate_response_video(prompt: str, llm_provider_video: str, video_file: Union[str, TextIO]) -> str:
    """
    多模态能力大模型
    """
    if llm_provider_video == "gemini":
        api_key = config.app.get("gemini_api_key")
        model_name = config.app.get("gemini_model_name")
        base_url = "***"
    else:
        raise ValueError(
            "llm_provider 未设置，请在 config.toml 文件中进行设置。"
        )

    if llm_provider_video == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=api_key, transport="rest")

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
        )

        try:
            response = model.generate_content([prompt, video_file])
            return response.text
        except Exception as err:
            return handle_exception(err)


def compress_video(input_path: str, output_path: str):
    """
    压缩视频文件
    Args:
        input_path: 输入视频文件路径
        output_path: 输出压缩后的视频文件路径
    """
    # 如果压缩后的视频文件已经存在，则直接使用
    if os.path.exists(output_path):
        logger.info(f"压缩视频文件已存在: {output_path}")
        return

    try:
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec='libx264', audio_codec='aac', bitrate="500k", audio_bitrate="128k")
    except subprocess.CalledProcessError as e:
        logger.error(f"视频压缩失败: {e}")
        raise


def generate_script(
    video_path: str, video_plot: str, video_name: str, language: str = "zh-CN", progress_callback=None
) -> str:
    """
    生成视频剪辑脚本
    Args:
        video_path: 视频文件路径
        video_plot: 视频剧情内容
        video_name: 视频名称
        language: 语言
        progress_callback: 进度回调函数

    Returns:
        str: 生成的脚本
    """
    try:
        # 1. 压缩视频
        compressed_video_path = f"{os.path.splitext(video_path)[0]}_compressed.mp4"
        compress_video(video_path, compressed_video_path)

        # 在关键步骤更新进度
        if progress_callback:
            progress_callback(15, "压缩完成")  # 例如,在压缩视频后

        # 2. 转录视频
        transcription = gemini_video_transcription(
            video_name=video_name,
            video_path=compressed_video_path,
            language=language,
            llm_provider_video=config.app["video_llm_provider"],
            progress_callback=progress_callback
        )
        if progress_callback:
            progress_callback(60, "生成解说文案...")  # 例如,在转录视频后

        # 3. 编写解说文案
        script = writing_short_play(video_plot, video_name, config.app["llm_provider"], count=300)

        # 在关键步骤更新进度
        if progress_callback:
            progress_callback(70, "匹配画面...")  # 例如,在生成脚本后

        # 4. 文案匹配画面
        if transcription != "":
            matched_script = screen_matching(huamian=transcription, wenan=script, llm_provider=config.app["video_llm_provider"])
            # 在关键步骤更新进度
            if progress_callback:
                progress_callback(80, "匹配成功")
            return matched_script
        else:
            return ""
    except Exception as e:
        handle_exception(e)
        raise


def gemini_video_transcription(video_name: str, video_path: str, language: str, llm_provider_video: str, progress_callback=None):
    '''
    使用 gemini-1.5-xxx 进行视频画面转录
    '''
    api_key = config.app.get("gemini_api_key")
    gemini.configure(api_key=api_key)

    prompt = """
    请转录音频，包括时间戳，并提供视觉描述，然后以 JSON 格式输出，当前视频中使用的语言为 %s。
    
    在转录视频时，请通过确保以下条件来完成转录：
    1. 画面描述使用语言: %s 进行输出。
    2. 同一个画面合并为一个转录记录。
    3. 使用以下 JSON schema:    
        Graphics = {"timestamp": "MM:SS-MM:SS"(时间戳格式), "picture": "str"(画面描述), "speech": "str"(台词，如果没有人说话，则使用空字符串。)}
        Return: list[Graphics]
    4. 请以严格的 JSON 格式返回数据，不要包含任何注释、标记或其他字符。数据应符合 JSON 语法，可以被 json.loads() 函数直接解析， 不要添加 ```json 或其他标记。
    """ % (language, language)

    logger.debug(f"视频名称: {video_name}")
    try:
        if progress_callback:
            progress_callback(20, "上传视频至 Google cloud")
        gemini_video_file = gemini.upload_file(video_path)
        logger.debug(f"视频 {gemini_video_file.name} 上传至 Google cloud 成功, 开始解析...")
        while gemini_video_file.state.name == "PROCESSING":
            gemini_video_file = gemini.get_file(gemini_video_file.name)
            if progress_callback:
                progress_callback(30, "上传成功, 开始解析")  # 更新进度为20%
        if gemini_video_file.state.name == "FAILED":
            raise ValueError(gemini_video_file.state.name)
        elif gemini_video_file.state.name == "ACTIVE":
            if progress_callback:
                progress_callback(40, "解析完成, 开始转录...")  # 更新进度为30%
            logger.debug("解析完成, 开始转录...")
    except ResumableUploadError as err:
        logger.error(f"上传视频至 Google cloud 失败, 用户的位置信息不支持用于该API; \n{traceback.format_exc()}")
        return False
    except FailedPrecondition as err:
        logger.error(f"400 用户位置不支持 Google API 使用。\n{traceback.format_exc()}")
        return False

    if progress_callback:
        progress_callback(50, "开始转录")
    try:
        response = _generate_response_video(prompt=prompt, llm_provider_video=llm_provider_video, video_file=gemini_video_file)
        logger.success("视频转录成功")
        logger.debug(response)
        print(type(response))
        return response
    except Exception as err:
        return handle_exception(err)


def generate_terms(video_subject: str, video_script: str, amount: int = 5) -> List[str]:
    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
Generate {amount} search terms for stock videos, depending on the subject of a video.

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.

## Output Example:
["search term 1", "search term 2", "search term 3","search term 4","search term 5"]

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            search_terms = json.loads(response)
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        logger.warning(f"failed to generate video terms: {str(e)}")
                        pass

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


def gemini_video2json(video_origin_name: str, video_origin_path: str, video_plot: str, language: str) -> str:
    '''
    使用 gemini-1.5-pro 进行影视解析
    Args:
        video_origin_name: str - 影视作品的原始名称
        video_origin_path: str - 影视作品的原始路径
        video_plot: str - 影视作品的简介或剧情概述

    Return:
        str - 解析后的 JSON 格式字符串
    '''
    api_key = config.app.get("gemini_api_key")
    model_name = config.app.get("gemini_model_name")

    gemini.configure(api_key=api_key)
    model = gemini.GenerativeModel(model_name=model_name)

    prompt = """
**角色设定：**  
你是一位影视解说专家，擅长根据剧情生成引人入胜的短视频解说文案，特别熟悉适用于TikTok/抖音风格的快速、抓人视频解说。

**任务目标：**  
1. 根据给定剧情，详细描述画面，重点突出重要场景和情节。  
2. 生成符合TikTok/抖音风格的解说，节奏紧凑，语言简洁，吸引观众。  
3. 解说的时候需要解说一段播放一段原视频，原视频一般为有台词的片段，原视频的控制有 OST 字段控制。
4. 结果输出为JSON格式，包含字段：  
   - "picture"：画面描述  
   - "timestamp"：画面出现的时间范围  
   - "narration"：解说内容
   - "OST": 是否开启原声（true / false）

**输入示例：**  
```text  
在一个���暗的小巷中，主角缓慢走进，四周静谧无声，只有远处隐隐传来猫的叫声。突然，背后出现一个神秘的身影。  
```  

**输出格式：**  
```json  
[  
    {  
        "picture": "黑暗的小巷，主角缓慢走入，四周安静，远处传来猫叫声。",  
        "timestamp": "00:00-00:17",  
        "narration": "静谧的小巷里，主角步步前行，气氛渐渐变得压抑。"  
        "OST": False  
    },  
    {  
        "picture": "神秘身影突然出现，紧张气氛加剧。",  
        "timestamp": "00:17-00:39",  
        "narration": "原声播放"  
        "OST": True  
    }  
]  
```  

**提示：**  
- 文案要简短有力，契合短视频平台用户的观赏习惯。  
- 保持强烈的悬念和情感代入，吸引观众继续观看。  
- 解说一段后播放一段原声，原声内容尽量和解说匹配。
- 文案语言为：%s  
- 剧情内容：%s (为空则忽略)  

""" % (language, video_plot)

    logger.debug(f"视频名称: {video_origin_name}")
    # try:
    gemini_video_file = gemini.upload_file(video_origin_path)
    logger.debug(f"上传视频至 Google cloud 成功: {gemini_video_file.name}")
    while gemini_video_file.state.name == "PROCESSING":
        import time
        time.sleep(1)
        gemini_video_file = gemini.get_file(gemini_video_file.name)
        logger.debug(f"视频当前状态(ACTIVE才可用): {gemini_video_file.state.name}")
    if gemini_video_file.state.name == "FAILED":
        raise ValueError(gemini_video_file.state.name)
    # except Exception as err:
    #     logger.error(f"上传视频至 Google cloud 失败, 请检查 VPN 配置和 APIKey 是否正确 \n{traceback.format_exc()}")
    #     raise TimeoutError(f"上传视频至 Google cloud 失败, 请检查 VPN 配置和 APIKey 是否正确; {err}")

    streams = model.generate_content([prompt, gemini_video_file], stream=True)
    response = []
    for chunk in streams:
        response.append(chunk.text)

    response = "".join(response)
    logger.success(f"llm response: \n{response}")

    return response


def writing_talk_show(video_plot, video_name, llm_provider):
    """
    影视解说（电影解说）
    """
    prompt = f"""
    **角色设定：**  
    你是一名有10年经验的播客访谈视频解说文案的创作者，
    下面是关于如何写解说文案的方法 {Method}，请认真阅读它，之后我会给你一个播客访谈类视频作品的名称，然后让你写一篇文案
    请根据方法撰写 《{video_name}》的访谈播客解说文案，《{video_name}》的大致内容如下: {video_plot}
    文案要符合以下要求:
    
    **任务目标：**  
    1. 文案字数在 1500字左右，严格要求字数，最低不得少于 1000字。
    2. 避免使用 markdown 格式输出文案。  
    3. 仅输出解说文案，不输出任何其他内容。
    4. 不要包含小标题，每个段落以 \n 进行分隔。
    """
    try:
        response = _generate_response(prompt, llm_provider)
        logger.success("解说文案生成成功")
        return response
    except Exception as err:
        return handle_exception(err)


def writing_short_play(video_plot: str, video_name: str, llm_provider: str, count: int = 500):
    """
    影视解说（短剧解说）
    """
    if not video_plot:
        raise ValueError("短剧的简介不能为空")
    if not video_name:
        raise ValueError("短剧名称不能为空")

    prompt = f"""
    **角色设定：**  
    你是一名有10年经验的短剧解说文案的创作者，
    下面是关于如何写解说文案的方法 {Method}，请认真阅读它，之后我会给你一部短剧作品的简介，然后让你写一篇解说文案
    请根据方法撰写 《{video_name}》的解说文案，《{video_name}》的大致剧情如下: {video_plot}
    文案要符合以下要求:

    **任务目标：**  
    1. 请严格要求文案字数, 字数控制在 {count} 字左右。
    2. 避免使用 markdown 格式输出文案。
    3. 仅输出解说文案，不输出任何其他内容。
    4. 不要包含小标题，每个段落以 \\n 进行分隔。
    """
    try:
        response = _generate_response(prompt, llm_provider)
        logger.success("解说文案生成成功")
        logger.debug(response)
        return response
    except Exception as err:
        return handle_exception(err)


def screen_matching(huamian: str, wenan: str, llm_provider: str):
    """
    画面匹配（一次性匹配）
    """
    if not huamian:
        raise ValueError("画面不能为空")
    if not wenan:
        raise ValueError("文案不能为空")

    prompt = """
    你是一名有10年经验的影视解说创作者，
    你的任务是根据视频转录脚本和解说文案，匹配出每段解说文案对应的画面时间戳, 结果以 json 格式输出。
    
    注意：
    转录脚本中 
        - timestamp: 表示视频时间戳
        - picture: 表示当前画面描述
        - speech": 表示当前视频中人物的台词
    
    转录脚本和文案（由 XML 标记<PICTURE></PICTURE>和 <COPYWRITER></COPYWRITER>分隔）如下所示：
    <PICTURE>
    %s
    </PICTURE>
    
    <COPYWRITER>
    %s
    </COPYWRITER>

    在匹配的过程中，请通过确保以下条件来完成匹配：
    - 使用以下 JSON schema:    
        script = {'picture': str, 'timestamp': str(时间戳), "narration": str, "OST": bool(是否开启原声)}
        Return: list[script]
    - picture: 字段表示当前画面描述，与转录脚本保持一致
    - timestamp: 字段表示某一段文案对应的画面的时间戳，不必和转录脚本的时间戳一致，应该充分考虑文案内容，匹配出与其描述最匹配的时间戳
        - 请注意，请严格的执行已经出现的画面不能重复出现，即生成的脚本中 timestamp 不能有重叠的部分。
    - narration: 字段表示需要解说文案，每段解说文案尽量不要超过30字
    - OST: 字段表示是否开启原声，即当 OST 字段为 true 时，narration 字段为空字符串，当 OST 为 false 时，narration 字段为对应的解说文案
    - 注意，在画面匹配的过程中，需要适当的加入原声播放，使得解说和画面更加匹配，请按照 1:1 的比例，生成原声和解说的脚本内容。
    - 注意，在时间戳匹配上，一定不能原样照搬“转录脚本”，应当适当的合并或者删减一些片段。
    - 注意，第一个画面一定是原声播放并且时长不少于 20 s，为了吸引观众，第一段一定是整个转录脚本中最精彩的片段。
    - 请以严格的 JSON 格式返回数据，不要包含任何注释、标记或其他字符。数据应符合 JSON 语法，可以被 json.loads() 函数直接解析， 不要添加 ```json 或其他标记。
    """ % (huamian, wenan)

    try:
        response = _generate_response(prompt, llm_provider)
        logger.success("匹配成功")
        logger.debug(response)
        return response
    except Exception as err:
        return handle_exception(err)


if __name__ == "__main__":
    # 1. 视频转录
    video_subject = "第二十条之无罪释放"
    video_path = "/Users/apple/Desktop/home/pipedream_project/downloads/jianzao.mp4"
    language = "zh-CN"
    gemini_video_transcription(
        video_name=video_subject,
        video_path=video_path,
        language=language,
        progress_callback=print,
        llm_provider_video="gemini"
    )

    # # 2. 解说文案
    # video_path = "/Users/apple/Desktop/home/NarratoAI/resource/videos/1.mp4"
    # # video_path = "E:\\projects\\NarratoAI\\resource\\videos\\1.mp4"
    # video_plot = """
    #     李自忠拿着儿子李牧名下的存折，去银行取钱给儿子救命，却被要求证明"你儿子是你儿子"。
    # 走投无路时碰到银行被抢劫，劫匪给了他两沓钱救命，李自忠却因此被银行以抢劫罪起诉，并顶格判处20年有期徒刑。
    # 苏醒后的李牧坚决为父亲做无罪辩护，面对银行的顶级律师团队，他一个法学院大一学生，能否力挽狂澜，创作奇迹？挥法律之利剑 ，持正义之天平！
    # """
    # res = generate_script(video_path, video_plot, video_name="第二十条之无罪释放")
    # # res = generate_script(video_path, video_plot, video_name="海岸")
    # print("脚本生成成功:\n", res)
    # res = clean_model_output(res)
    # aaa = json.loads(res)
    # print(json.dumps(aaa, indent=2, ensure_ascii=False))
