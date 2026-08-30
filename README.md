# 建筑高度估计

本目录将建筑估高方法与当前 PS 输入放在同一类别下：

- `building_height_estimation_pixel_offset/`：基于候选屋顶高程投影与影像结构匹配的像素偏移法。
- `building_height_estimation_roof_only/`：屋顶证据搜索及可观测性审计流程。
- `ps_triangle_height_estimation/`：PS 屋顶/墙面归属、三角面反算、稳健平差与高层顶部恢复。
- `ps_coordinates_current/`：当前推荐的 23,178 个 PS 点输入包。

三类估高流程共享 `geocoding/data/`，但各自保持独立的 `config*.json`、`code/` 和 `results/`。观测方程、质量等级和样本数不得跨方法混用。

当前 PS 三角面流程可从工作区根目录运行：

```bash
bash height_estimation/ps_triangle_height_estimation/run.sh
bash height_estimation/ps_triangle_height_estimation/run_height_estimation.sh
```

详细方法口径、推荐结果和高程基准见根目录 `agent.md`。

## 历史依赖说明

`building_height_estimation_pixel_offset/config.json` 和 `building_height_estimation_roof_only/config*.json` 中的部分 `../building_height_estimation/` 字段指向早期已不在本工作区的严格候选几何代码与对照表。这些字段不是本次迁移前的根路径，因此作为历史复现依赖保留，不伪造替代数据。当前路径完整的入口是 PS 三角面流程和 `building_height_estimation_pixel_offset/siwei_gaojing_reproduction/`；重跑早期 TerraSAR-X 像素偏移/屋顶流程前，需先恢复该历史依赖包。
