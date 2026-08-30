# 四维高景建筑高度估计——清晰版全流程

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
7. 写入建筑矢量：当前435/1028栋有最终高度。
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
