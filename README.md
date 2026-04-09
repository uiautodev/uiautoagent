# AI RPA - 基于视觉模型的UI元素检测器

使用 AI 视觉模型检测截图中的 UI 元素并返回边界框坐标，可用于自动化 UI 测试、RPA 等场景。

## 特性

- 🎯 基于视觉模型的元素定位，无需依赖 DOM 结构
- 📏 返回精确的边界框坐标 (bbox)
- 💭 包含 AI 的思考过程，便于调试和验证
- 🖼️ 支持自动绘制检测结果到图片
- 🔧 兼容 OpenAI API 格式，支持多种视觉模型

## 安装

```bash
# 使用 uv 安装依赖
uv sync
```

## 配置

复制 `.env.example` 为 `.env` 并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
BASE_URL=https://api.openai.com/v1
API_KEY=sk-xxx
MODEL_NAME=gpt-4o
REQUEST_TIMEOUT=30
```

**支持的视觉模型：**
- OpenAI: `gpt-4o`, `gpt-4o-mini`
- 兼容 OpenAI API 的其他视觉模型

## 使用方法

### 命令行

```bash
# 使用默认配置
uv run main.py

# 指定图片和查询
uv run main.py -i screenshot.png -q "登录按钮"

# 查看帮助
uv run main.py --help
```

### Python API

```python
from bbox_detector import detect_element, draw_bbox

# 检测元素
result = detect_element("screenshot.png", "登录按钮")

if result.found:
    print(f"找到: {result.description}")
    print(f"坐标: {result.bbox}")
    print(f"中心点: {result.bbox.center}")
    print(f"思考: {result.thought}")

# 绘制检测结果
draw_bbox("screenshot.png", result, "output.png")
```

### 返回结果

```python
class DetectionResult:
    found: bool           # 是否找到元素
    bbox: BBox | None     # 边界框坐标
    description: str      # 元素描述
    thought: str          # AI 的思考过程

class BBox:
    x1: int               # 左上角 X 坐标
    y1: int               # 左上角 Y 坐标
    x2: int               # 右下角 X 坐标
    y2: int               # 右下角 Y 坐标
    center: tuple[int, int]  # 中心点坐标
    width: int            # 宽度
    height: int           # 高度
```

## 项目结构

```
.
├── bbox_detector.py    # 核心检测器模块
├── main.py             # 命令行入口
├── .env.example        # 环境变量示例
├── sample.png          # 示例截图
└── result.png          # 检测结果输出（自动生成）
```

## 示例

### 输入图片 (sample.png)

![sample.png](sample.png)

### 检测结果 (result.png)

![result.png](result.png)

## 注意事项

- 模型必须支持视觉能力 (vision-capable)
- 坐标基于实际图片尺寸，会自动缩放
- 建议使用清晰的截图以获得最佳结果
