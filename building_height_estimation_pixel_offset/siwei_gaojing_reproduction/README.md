# 四维高景像素偏移建筑估高复现

## 主入口

整理后的唯一主入口为 [`清晰版全流程/README.md`](清晰版全流程/README.md)。
该目录按输入审计、核心过程图、最终表格和GIS成果分组。下方的 `results/picall/正式图件`
仅作为原16图算法审计归档，不再作为日常结果入口。

本目录复用上级工程 `results/picall/正式图件` 的 16 图算法链，但输入替换为
`data/四维高景`。为满足原算法“多景影像必须位于同一雷达像素网格”的前提，正式
复现使用两景同为右视降轨、同为 SP_527 的 SLC：

- 2026-06-24，SVN2-03，SP_527，降轨（主影像）；
- 2026-06-16，SVN2-05，SP_527，降轨（辅影像）。

2026-04-12 为升轨，2026-03-28 为 SP_506 且入射角明显不同，不能直接与主影像逐
像素取中位数，因此保留在 `inputs/input_manifest.json` 的排除审计中。

## 运行

```bash
PY=/home/u/geocoding/tongji_sbas/.venv/bin/python
$PY code/prepare_siwei_inputs.py
$PY code/run_reproduction.py
```

输出写入本目录的 `results/picall/正式图件`、`results/tables`、`results/vectors` 和
`results/picall/过程图件`，不会覆盖上级工程已有结果。

最终可直接在 GIS 中使用的建筑面成果为
`results/vectors/final_building_heights_wgs84.gpkg`：`all_buildings` 图层含全部
1028 栋建筑及质量字段，`accepted_heights` 图层仅含通过筛选的高度。其他部分
GeoPackage 图层表示 SAR 影像行列像素坐标，因此有意不赋地理 CRS。

## 几何定义

- 建筑矢量：WGS 84（运行前显式检查并转换为 EPSG:4326）；
- 建筑底面：吴淞高程 4 m；
- 矢量 `height`：解释为离地建筑高度；
- 屋顶绝对高程：`4 m + height`；
- 最终建筑高度：`反演屋顶绝对高程 - 4 m`；
- GAMMA 正投影高度：暂用 `吴淞高程 + GAMMA EGM96` 作为 WGS84 椭球高代理；
- 主影像裁剪：原始 SLC 距离向 `[22351, 29077)`、方位向 `[0, 4703)`；
- 算法网格：将 6726×4703 裁剪重采样为 900×630；
- 2026-06-16 影像通过两景 GAMMA 地面控制点投影拟合仿射关系，重采样到主影像网格。

此处的 EGM96 转换不是经测量标定的吴淞高程转换，最终高度精度仍需外部真实建筑
高度验证。
