"""示例：检测截图中的UI元素"""

import argparse
from bbox_detector import detect_element, draw_bbox


def main():
    parser = argparse.ArgumentParser(
        description="检测截图中的UI元素",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--image", default="sample.png", help="图片路径")
    parser.add_argument(
        "-q",
        "--query",
        default="弹窗关闭按钮",
        help="要查找的元素描述",
    )
    args = parser.parse_args()

    image = args.image
    queries = [args.query]

    results = []
    for q in queries:
        print(f"\n🔍 查找: {q}")
        result = detect_element(image, q)
        results.append(result)
        if result.found and result.bbox:
            print(f"  ✅ {result.description}")
            print(f"  📐 {result.bbox}")
            print(f"  🎯 中心点: {result.bbox.center}")
        else:
            print(f"  ❌ 未找到")
        if result.thought:
            print(f"  💭 思考: {result.thought}")

    # 绘制标注图
    if results and results[0].found and results[0].bbox:
        draw_bbox(image, results[0], "result.png")
        print("\n已保存标注结果到 result.png")


if __name__ == "__main__":
    main()
