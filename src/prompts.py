"""
Prompt templates used by the three-stage pipeline.

Design notes:
- Phase 1 and Phase 2 remain controlled so that descriptions and questions stay usable.
- Phase 3 is pure distillation: image + question go directly to the teacher, and we keep
  the teacher's natural answer without adding behavioral guardrails or output format rules.
"""

DESCRIPTION_PROMPT = """你是一位专业的遥感图像解译人员。请对这张遥感图像生成受控描述。

## 描述要求
1. 只描述图像中清晰可见的内容，不要猜测图像外信息。
2. 覆盖以下方面：
   - 场景类型：城市、乡村、农田、水体、植被、工业区、灾后区域等
   - 主要地物：建筑、道路、植被、水体、车辆、裸地、人工设施等
   - 空间关系：相邻、位于某方向、沿道路分布、被道路分隔等
   - 可计数对象：只统计清晰可见的对象，不强行计入遮挡或模糊目标
   - 不确定内容：无法确认的用途、类型或数量需要明确标注
3. 不输出行动建议、灾害成因、治理方案、精确坐标或精确面积。
4. 如果目标不清晰，可以直接说明图像中难以辨认。

请用清晰、客观的语言分点描述，不要堆砌术语。"""


QUESTION_GENERATION_PROMPT = """基于这张遥感图像生成 VQA 问题。
下面的描述仅供参考，可能有误；请以图像实际内容为准。

## 图像描述（仅供参考，可能有误）
{description}

## 本图建议生成的问题类型与数量
请按下面的清单生成问题（每类按指定数量）：
{assigned_plan}

各类型含义：
- existence: 是否存在某类地物，正反都要问
- location: 某地物位于图像哪个区域
- spatial_relation: 两个地物之间的相对位置或邻接关系
- attribute: 颜色、形状、屋顶类型、材质或纹理
- counting: 清晰可见、数量较少、可准确点数的目标数量；遮挡、密集或模糊场景不适合
- unanswerable: 仅凭该图像无法确认的问题，如用途、材质、年代、权属等
- ambiguous: 图像中存在多个合理解释、难以唯一确定的问题

## 输出格式
每行一个问题，格式必须为：
类型|问题内容

例如：
existence|图像中是否存在道路？
ambiguous|图像中央那栋大跨度建筑是仓库还是厂房？

注意：
1. 尽量贴合上面的类型清单；但若某个类型确实不适合本图，可以跳过，不要硬凑。
2. 问题简短明确，只问一个具体方面；除 unanswerable 和 ambiguous 外，应能仅凭图像回答。
3. existence 问题要正反均衡，不要只问图里已有的对象。
4. 同类问题不要用相同句式起手，也不要反复问同一对象。
5. 只输出问题列表，不要输出解释性文字。"""


ANSWER_GENERATION_PROMPT = """请回答这个遥感 VQA 问题，并按指定格式输出分析与答案。

## 问题
{question}

## 问题类型
{question_type}

## 图像描述（仅供参考，可能有误）
{description}

## 核心原则
1. 一切以图像实际内容为准。描述仅作参考；当描述与图像冲突时，以图像为准。
2. 只依据图像中可见内容回答，不推断图像外信息。
3. 不给出行政、工程或治理建议，不编造坐标、面积、距离或精确数量。
4. 若图像信息不足以判断，可以直接说明无法根据图像确认。

## analysis 写法要求
1. analysis 是“可见证据链”，不是任务拆解或格式说明。
2. 步数按问题复杂度决定：简单问题 1 步即可，复杂问题最多 3 步。
3. 每步先点出要点再写内容，每步 1-2 句，整体不超过 180 个中文字符。

## 输出格式
<analysis>
1. ...
2. ...（按需，可省略）
3. ...（按需，可省略）
</analysis>
<answer>最终答案，简洁直接</answer>

## 各类型分析重点
- existence: 观察目标是否可见后给出存在性结论
- attribute: 观察颜色、形状、纹理或屋顶特征后给出属性结论
- location: 判断目标相对方位，使用左上、右下、中部等方位词
- spatial_relation: 分别定位两个目标，再比较其相对位置或邻接关系
- counting: 统计清晰可见目标；若遮挡或模糊影响点数，可说明难以精确
- unanswerable: 简要说明图像无法支持判断
- ambiguous: 列出主要可能性，并说明无法唯一确定

注意：只输出指定格式的内容，不要输出开场白或解释性文字。"""


ANSWER_DISTILL_PROMPT = """{question}"""


ANSWER_DIRECT_PROMPT = """请直接回答这个遥感 VQA 问题，只给出最终答案，不要写分析过程。

## 问题
{question}

## 问题类型
{question_type}

## 图像描述（仅供参考，可能有误）
{description}

## 要求
1. 一切以图像实际内容为准；描述与图像冲突时以图像为准。
2. 只依据图像可见内容回答，不推断图像外信息。
3. 答案简洁直接，一般不超过 40 字。
4. 如果图像信息不足，可以直接说明无法根据图像确认。

## 输出格式
<answer>最终答案，简洁直接</answer>

只输出 <answer>...</answer>，不要输出任何其他内容。"""
