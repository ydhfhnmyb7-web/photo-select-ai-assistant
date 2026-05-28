from __future__ import annotations


MANUAL_SELECTED = "精修候选"
MANUAL_PENDING = "备选"
MANUAL_REJECTED = "废片"
MANUAL_REVIEW = "人工复核"
MANUAL_RETOUCH = "可交付"
MANUAL_CROP = "重复"
MANUAL_PORTFOLIO = "作品集候选"
MANUAL_PRACTICE = "练习片"
MANUAL_CATEGORIES = [
    MANUAL_SELECTED,
    MANUAL_RETOUCH,
    MANUAL_PENDING,
    MANUAL_CROP,
    MANUAL_REJECTED,
    MANUAL_PORTFOLIO,
    MANUAL_PRACTICE,
    MANUAL_REVIEW,
]

LEGACY_MANUAL_CATEGORY_MAP = {
    "精选": MANUAL_SELECTED,
    "待定": MANUAL_PENDING,
    "可用待修": MANUAL_RETOUCH,
    "可用待裁切": MANUAL_CROP,
}

PRIMARY_REVIEW = "人工复核"
PRIMARY_DUPLICATE = "相似重复待选"
PRIMARY_REJECT = "废片疑似"

PRIMARY_CATEGORIES = [
    "单人写真",
    "双人写真",
    "多人写真",
    "婚纱照",
    "全家福",
    "亲子照",
    "儿童写真",
    "中老年写真",
    "商务会议合影",
    "活动纪实",
    "证件/形象照",
    "风景/环境",
    PRIMARY_DUPLICATE,
    PRIMARY_REJECT,
    PRIMARY_REVIEW,
]

SECONDARY_CATEGORIES = {
    "单人写真": ["年轻女生", "年轻男生", "中年女性", "中年男性", "老年女性", "老年男性", "儿童", "其他单人"],
    "双人写真": ["情侣", "闺蜜姐妹", "兄弟朋友", "母女", "父子", "其他双人"],
    "多人写真": ["朋友合影", "闺蜜团", "兄弟团", "小组合影", "其他多人"],
    "婚纱照": ["单人婚纱", "双人婚纱", "中式婚纱", "外景婚纱", "棚拍婚纱"],
    "全家福": ["三口之家", "多人家庭", "长辈合影", "其他家庭合影"],
    "亲子照": ["母子/母女", "父子/父女", "双亲亲子", "其他亲子"],
    "商务会议合影": ["单人商务形象", "小组合影", "大合影", "会议现场"],
}

QUALITY_CLEAR = "清晰"
QUALITY_SLIGHT_BLURRY = "轻微模糊"
QUALITY_SEVERE_BLURRY = "严重模糊"
QUALITY_BLURRY = QUALITY_SLIGHT_BLURRY
QUALITY_SLIGHT_OVEREXPOSED = "轻微过曝"
QUALITY_SEVERE_OVEREXPOSED = "严重过曝"
QUALITY_SLIGHT_UNDEREXPOSED = "轻微欠曝"
QUALITY_SEVERE_UNDEREXPOSED = "严重欠曝"
QUALITY_EYES_CLOSED = "疑似闭眼"
QUALITY_BAD_EXPRESSION = "表情不佳"
QUALITY_EXPOSURE = "曝光异常"
QUALITY_BACKGROUND_ISSUE = "背景穿帮"
QUALITY_EDGE_ISSUE = "边缘穿帮"
QUALITY_RE_CROP = "可二次裁切"
QUALITY_LOOSE_COMPOSITION = "构图偏松"
QUALITY_TIGHT_COMPOSITION = "构图偏紧"
QUALITY_NATURAL_EXPRESSION = "表情自然"
QUALITY_FAILED_EXPRESSION = "表情失败"
QUALITY_POSE_OK = "姿态可用"
QUALITY_POSE_FAILED = "姿态失败"
QUALITY_COMPOSITION = QUALITY_BACKGROUND_ISSUE
QUALITY_DUPLICATE = "重复相似"
QUALITY_KEEP = "推荐保留"
QUALITY_USABLE_RETOUCH = "可用待修"
QUALITY_USABLE_CROP = "可用待裁切"
QUALITY_REVIEW = "人工复核"
QUALITY_CLEAR_REJECT = "明确废片"
QUALITY_UNCERTAIN = "模型不确定"
QUALITY_CONFLICT = "AI与用户偏好冲突"
QUALITY_LEARNING_SAMPLE = "已加入学习样本"

QUALITY_TAGS = [
    QUALITY_CLEAR,
    QUALITY_SLIGHT_BLURRY,
    QUALITY_SEVERE_BLURRY,
    QUALITY_SLIGHT_OVEREXPOSED,
    QUALITY_SEVERE_OVEREXPOSED,
    QUALITY_SLIGHT_UNDEREXPOSED,
    QUALITY_SEVERE_UNDEREXPOSED,
    QUALITY_BACKGROUND_ISSUE,
    QUALITY_EDGE_ISSUE,
    QUALITY_RE_CROP,
    QUALITY_LOOSE_COMPOSITION,
    QUALITY_TIGHT_COMPOSITION,
    QUALITY_NATURAL_EXPRESSION,
    QUALITY_FAILED_EXPRESSION,
    QUALITY_EYES_CLOSED,
    QUALITY_POSE_OK,
    QUALITY_POSE_FAILED,
    QUALITY_DUPLICATE,
    QUALITY_KEEP,
    QUALITY_USABLE_RETOUCH,
    QUALITY_USABLE_CROP,
    QUALITY_REVIEW,
    QUALITY_CLEAR_REJECT,
    QUALITY_UNCERTAIN,
    QUALITY_CONFLICT,
    QUALITY_LEARNING_SAMPLE,
]


def category_path(primary: str, secondary: str = "") -> str:
    if secondary:
        return f"{primary}/{secondary}"
    return primary or PRIMARY_REVIEW


def split_category(value: str) -> tuple[str, str]:
    parts = [part for part in (value or "").split("/") if part]
    if not parts:
        return PRIMARY_REVIEW, ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def manual_category_path(manual_category: str) -> str:
    manual_category = LEGACY_MANUAL_CATEGORY_MAP.get(manual_category, manual_category)
    if manual_category == MANUAL_SELECTED:
        return "精修候选"
    if manual_category == MANUAL_RETOUCH:
        return "可交付"
    if manual_category == MANUAL_PENDING:
        return "备选"
    if manual_category == MANUAL_CROP:
        return "重复"
    if manual_category == MANUAL_REJECTED:
        return "废片"
    if manual_category == MANUAL_PORTFOLIO:
        return "作品集候选"
    if manual_category == MANUAL_PRACTICE:
        return "练习片"
    if manual_category == MANUAL_REVIEW:
        return "人工复核"
    return "备选"
