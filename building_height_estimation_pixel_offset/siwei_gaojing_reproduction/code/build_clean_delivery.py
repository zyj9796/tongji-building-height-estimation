"""Build a compact, unambiguous delivery package from completed results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans", "Arial"],
    "svg.fonttype": "none",
})


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "清晰版全流程"


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def draw_overview(path: Path) -> None:
    stages = [
        ("01 数据审计", "筛选同轨同波束两景\n检查建筑CRS与几何"),
        ("02 底面投影", "吴淞高程4 m\nGAMMA严格投影"),
        ("03 建立尺子", "逐栋计算高程方向\n像素偏移/米"),
        ("04 顶面定位", "扩长二维搜索\n轮廓、边缘、内外对比"),
        ("05 高程迭代", "候选高度逐次GAMMA重投影\n0.1 m细化"),
        ("06 质量仲裁", "两景一致、非边界\n残差、分支、重叠冲突"),
        ("07 高度成果", "H = Zroof − 4 m\n写入建筑矢量"),
        ("08 像素赋高", "后续独立步骤\n侧面线性、顶面常高"),
    ]
    colors = ["#E8F1FA", "#DDF3F0", "#FFF0C9", "#FDE2D8", "#E9E0F5", "#F4E4EF", "#DDEED5", "#EEEEEE"]
    fig, ax = plt.subplots(figsize=(14.2, 8.2))
    ax.set_xlim(0, 14.2); ax.set_ylim(0, 8.2); ax.axis("off")
    positions = [(0.6, 5.2), (4.0, 5.2), (7.4, 5.2), (10.8, 5.2),
                 (10.8, 2.4), (7.4, 2.4), (4.0, 2.4), (0.6, 2.4)]
    for i, ((title, body), (x, y), color) in enumerate(zip(stages, positions, colors)):
        box = FancyBboxPatch((x, y), 2.75, 1.45, boxstyle="round,pad=0.08,rounding_size=0.08",
                             facecolor=color, edgecolor="#405060", linewidth=1.0)
        ax.add_patch(box)
        ax.text(x + 0.18, y + 1.05, title, fontsize=11, fontweight="bold", va="center")
        ax.text(x + 0.18, y + 0.52, body, fontsize=9, va="center", linespacing=1.4)
        if i < len(stages) - 1:
            x2, y2 = positions[i + 1]
            if i < 3:
                start, end = (x + 2.78, y + 0.72), (x2 - 0.04, y2 + 0.72)
            elif i == 3:
                start, end = (x + 1.38, y - 0.04), (x2 + 1.38, y2 + 1.50)
            else:
                start, end = (x - 0.04, y + 0.72), (x2 + 2.78, y2 + 0.72)
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                         linewidth=1.1, color="#405060"))
    ax.text(0.6, 7.65, "四维高景SAR建筑高度估计：整理后的单一流程", fontsize=18, fontweight="bold")
    ax.text(0.6, 7.18, "唯一高度定义：底面 Zbase=4 m（吴淞）｜屋顶 Zroof=4 m+H｜建筑高度 H=Zroof−Zbase",
            fontsize=11, color="#263746")
    ax.text(0.6, 1.42, "当前已完成：01–07", fontsize=10, fontweight="bold", color="#26734D")
    ax.text(0.6, 1.02, "08 像素赋高不参与建筑高度估计：先确定屋顶与底面，再沿侧面尺子线性插值；顶面像素赋同一屋顶高程。",
            fontsize=9.5)
    ax.text(0.6, 0.52, "注意：吴淞→WGS84椭球高目前采用 GAMMA EGM96 代理转换，尚未由实测控制点完成外部绝对精度验证。",
            fontsize=9, color="#8B3A2B")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for subdir in ("01_输入审计", "02_核心过程图", "03_最终表格", "04_GIS成果", "05_运行说明"):
        (OUT / subdir).mkdir(parents=True, exist_ok=True)

    draw_overview(OUT / "00_全流程总览.svg")
    copy(ROOT / "inputs/input_manifest.json", OUT / "01_输入审计/input_manifest.json")
    copy(ROOT / "config.json", OUT / "01_输入审计/config.json")

    figures = {
        ROOT / "results/clean_workflow/02_base_projection/01_图件_459712912978.svg": "01_底面4m_GAMMA投影.svg",
        ROOT / "results/picall/正式图件/06_全部建筑矢量高度投影.svg": "02_屋顶先验_4m加建筑高度.svg",
        ROOT / "results/picall/正式图件/08_合成孔径雷达建筑特征增强.svg": "03_SAR顶面轮廓特征增强.svg",
        ROOT / "results/picall/正式图件/12_纯影像特征局部配准.svg": "04_二维扩窗顶面定位.svg",
        ROOT / "results/picall/正式图件/15_数量质量联合配准.svg": "05_GAMMA严格迭代与质量仲裁.svg",
        ROOT / "results/picall/正式图件/16_数量质量联合建筑高度图.svg": "06_最终建筑高度图.svg",
        ROOT / "results/picall/正式图件/14_纯影像特征配准审计.svg": "07_顶面匹配质量审计.svg",
    }
    for source, name in figures.items():
        copy(source, OUT / "02_核心过程图" / name)

    raw = pd.read_csv(ROOT / "results/tables/joint_quantity_quality_building_heights.csv")
    clean = raw.rename(columns={
        "final_accepted": "accepted",
        "final_height_m": "building_height_m",
        "final_roof_elevation_m": "roof_elevation_wusong_m",
        "base_elevation_m": "base_elevation_wusong_m",
        "final_confidence": "confidence",
        "final_source": "source",
        "branch_height_agreement_m": "branch_agreement_m",
        "final_score_margin": "score_margin",
    })[["clean_id", "accepted", "base_elevation_wusong_m", "roof_elevation_wusong_m",
        "building_height_m", "confidence", "source", "branch_agreement_m", "score_margin"]]
    clean.to_csv(OUT / "03_最终表格/building_heights.csv", index=False)

    dictionary = pd.DataFrame([
        ("clean_id", "建筑唯一编号", "-"), ("accepted", "是否通过最终质量控制", "0/1"),
        ("base_elevation_wusong_m", "建筑底面吴淞高程，固定4 m", "m"),
        ("roof_elevation_wusong_m", "反演屋顶吴淞高程", "m"),
        ("building_height_m", "离地建筑高度=屋顶高程−底面高程", "m"),
        ("confidence", "high/medium/low/supplemental/none", "-"),
        ("source", "严格GAMMA迭代或混合分支来源", "-"),
        ("branch_agreement_m", "严格与混合分支高度差", "m"),
        ("score_margin", "最佳匹配相对次优匹配的分数间隔", "-"),
    ], columns=["field", "meaning", "unit"])
    dictionary.to_csv(OUT / "03_最终表格/字段说明.csv", index=False)
    copy(ROOT / "results/tables/joint_quantity_quality_optimization_summary.json",
         OUT / "03_最终表格/summary.json")

    buildings = gpd.read_file(ROOT / "results/vectors/final_building_heights_wgs84.gpkg", layer="all_buildings")
    attributes = clean.set_index("clean_id")
    drop = [c for c in raw.columns if c != "clean_id" and c in buildings.columns]
    buildings = buildings.drop(columns=drop).join(attributes, on="clean_id")
    if buildings.crs is None:
        raise RuntimeError("final geographic buildings have no CRS")
    buildings = buildings.to_crs(4326)
    if not buildings.geometry.is_valid.all():
        buildings.geometry = buildings.geometry.make_valid()
    accepted = buildings[buildings.accepted == 1].copy()
    gpkg = OUT / "04_GIS成果/building_heights_WGS84.gpkg"
    if gpkg.exists(): gpkg.unlink()
    buildings.to_file(gpkg, layer="all_buildings", driver="GPKG")
    accepted.to_file(gpkg, layer="accepted_heights", driver="GPKG")

    workflow = {
        "completed_stages": [1, 2, 3, 4, 5, 6, 7],
        "pending_optional_stage": 8,
        "base_definition": "Wusong elevation 4 m",
        "roof_definition": "Wusong roof elevation = 4 m + building height",
        "height_definition": "building height = roof elevation - 4 m",
        "buildings": int(len(clean)), "accepted": int(clean.accepted.sum()),
        "external_accuracy_validated": False,
    }
    (OUT / "03_最终表格/workflow_status.json").write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# 四维高景建筑高度估计——清晰版全流程

主入口只有三个：`00_全流程总览.svg`、`03_最终表格/building_heights.csv`、
`04_GIS成果/building_heights_WGS84.gpkg`。旧的16张PICALL图是算法审计归档，不再作为流程入口。

## 唯一高度定义

- 建筑底面：吴淞高程 `Zbase = 4 m`；
- 矢量 `height`：离地建筑高度，不是绝对高程；
- 屋顶先验：`Zroof,prior = 4 m + height`；
- 最终高度：`H = Zroof,estimated - 4 m`。

GAMMA接收投影高度前，代码用 `吴淞高程 + GAMMA EGM96起伏` 转为WGS84椭球高代理。

## 单一计算路线

1. 数据审计：只使用2026-06-24与2026-06-16两景同轨SP_527降轨数据。
2. 底面投影：1028栋全部以吴淞4 m逐顶点执行GAMMA严格投影。
3. 尺子建立：逐栋计算底面到不同屋顶高程的SAR位移方向和像素每米关系。
4. 顶面定位：扩大二维搜索范围，使用轮廓连续性、方向边缘、内外对比和强散射点群匹配。
5. 严格反演：线性尺子仅给初值，每个候选高度重新调用GAMMA，最后以0.1 m步长细化。
6. 质量仲裁：拒绝搜索边界解、两景不一致解、弱轮廓解和重复占用同一屋顶的解。
7. 写入建筑矢量：当前{int(clean.accepted.sum())}/{len(clean)}栋有最终高度。
8. SAR像素赋高：属于后续独立步骤；侧面沿底面—顶面方向线性变化，顶面像素使用同一屋顶高程。

## 复现命令

```bash
PY=/home/u/geocoding/tongji_sbas/.venv/bin/python
$PY code/prepare_siwei_inputs.py
$PY code/run_reproduction.py
$PY code/build_clean_delivery.py
```

## 精度边界

当前只完成内部几何和匹配质量控制，没有外部实测高度，因此不能把置信等级解释为RMSE或绝对精度。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "05_运行说明/原16图归档位置.txt").write_text(str(ROOT / "results/picall/正式图件") + "\n", encoding="utf-8")
    print(json.dumps({"delivery": str(OUT), "figures": len(figures) + 1, "accepted": int(clean.accepted.sum()),
                      "crs": str(buildings.crs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
