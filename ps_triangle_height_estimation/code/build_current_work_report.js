const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  Math: DocxMath,
  MathRun,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "../..");
const WORKSPACE_ROOT = path.resolve(ROOT, "..");
const RESULT_ROOT = path.join(
  ROOT,
  "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current",
);
const HIGH_ROOT = path.join(RESULT_ROOT, "highrise_optimized");
const OUTPUT = path.join(
  WORKSPACE_ROOT,
  "output/同济校区PS三角面建筑估高方法报告/同济校区PS三角面建筑估高方法原理报告.docx",
);
const BASE = JSON.parse(
  fs.readFileSync(path.join(RESULT_ROOT, "summary.json"), "utf8"),
);
const HIGH = JSON.parse(
  fs.readFileSync(path.join(HIGH_ROOT, "summary.json"), "utf8"),
);
const MAP = JSON.parse(
  fs.readFileSync(
    path.join(HIGH_ROOT, "final_mapping/mapping_summary.json"),
    "utf8",
  ),
);

const PAGE_WIDTH = 11906;
const PAGE_HEIGHT = 16838;
const CONTENT_WIDTH = 9600;
const FONT = "Noto Sans CJK SC";
const BLACK = "000000";
const BLUE = BLACK;
const CYAN = BLACK;
const GRAY = BLACK;
const DARK = BLACK;
const PALE = "FFFFFF";
const PALE_ORANGE = "FFFFFF";
const none = { style: BorderStyle.NONE, size: 0, color: BLACK };
const thin = { style: BorderStyle.SINGLE, size: 4, color: BLACK };
const thick = { style: BorderStyle.SINGLE, size: 8, color: BLACK };
const threeLineHeader = { top: thick, bottom: thin, left: none, right: none };
const threeLineBody = { top: none, bottom: none, left: none, right: none };
const threeLineLast = { top: none, bottom: thick, left: none, right: none };

function run(text, options = {}) {
  return new TextRun({
    text,
    font: options.font || FONT,
    size: options.size || 21,
    color: BLACK,
    bold: Boolean(options.bold),
    italics: Boolean(options.italics),
    break: options.break,
  });
}

function para(text, options = {}) {
  return new Paragraph({
    alignment: options.alignment || AlignmentType.JUSTIFIED,
    spacing: {
      before: options.before || 0,
      after: options.after === undefined ? 120 : options.after,
      line: options.line || 340,
    },
    indent: options.indent === false ? undefined : { firstLine: 420 },
    keepNext: Boolean(options.keepNext),
    children: Array.isArray(text) ? text : [run(text, options)],
  });
}

function note(title, text) {
  return new Paragraph({
    spacing: { before: 140, after: 180, line: 320 },
    indent: { left: 260, right: 260 },
    border: {
      top: { style: BorderStyle.SINGLE, size: 4, color: BLACK, space: 5 },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: BLACK, space: 5 },
    },
    children: [
      run(`${title}：`, { bold: true }),
      run(text),
    ],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    pageBreakBefore: true,
    spacing: { before: 0, after: 260 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 6, color: BLACK, space: 6 },
    },
    children: [run(text, { bold: true, size: 32 })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 260, after: 160 },
    keepNext: true,
    children: [run(text, { bold: true, size: 26 })],
  });
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 100 },
    keepNext: true,
    children: [run(text, { bold: true, size: 23 })],
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "report-bullets", level },
    spacing: { after: 80, line: 320 },
    children: [run(text)],
  });
}

function numbered(text, reference = "report-steps") {
  return new Paragraph({
    numbering: { reference, level: 0 },
    spacing: { after: 90, line: 320 },
    children: [run(text)],
  });
}

function equation(formula, number) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 160, line: 360 },
    keepNext: true,
    children: [
      run(formula, { font: "Cambria Math", size: 21 }),
      run(`    (${number})`, { font: "Cambria Math", size: 20 }),
    ],
  });
}

function cell(text, width, options = {}) {
  const cellBorders = options.header
    ? threeLineHeader
    : options.last
      ? threeLineLast
      : threeLineBody;
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: cellBorders,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 100, bottom: 100, left: 130, right: 130 },
    children: [
      new Paragraph({
        alignment: options.alignment || AlignmentType.LEFT,
        spacing: { after: 0, line: 280 },
        children: [
          run(String(text), {
            bold: Boolean(options.header || options.bold),
            size: options.size || 19,
          }),
        ],
      }),
    ],
  });
}

function table(headers, rows, widths) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        cantSplit: true,
        tableHeader: true,
        children: headers.map((x, i) => cell(x, widths[i], { header: true })),
      }),
      ...rows.map(
        (row, r) =>
          new TableRow({
            cantSplit: true,
            children: row.map((x, i) =>
              cell(x, widths[i], {
                last: r === rows.length - 1,
                alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
              }),
            ),
          }),
      ),
    ],
  });
}

function pngSize(file) {
  const buffer = fs.readFileSync(file);
  return {
    width: buffer.readUInt32BE(16),
    height: buffer.readUInt32BE(20),
  };
}

function figure(relativePath, caption, options = {}) {
  const file = path.join(ROOT, relativePath);
  const original = pngSize(file);
  const maxWidth = options.maxWidth || 620;
  const maxHeight = options.maxHeight || 690;
  const scale = Math.min(
    maxWidth / original.width,
    maxHeight / original.height,
  );
  const width = Math.round(original.width * scale);
  const height = Math.round(original.height * scale);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 90 },
      keepNext: true,
      keepLines: true,
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(file),
          transformation: { width, height },
          altText: {
            title: caption,
            description: caption,
            name: path.basename(file),
          },
        }),
      ],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 180, line: 290 },
      keepNext: Boolean(options.keepNext),
      children: [run(caption, { size: 18, color: GRAY })],
    }),
  ];
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function contentsTable() {
  const rows = [
    ["摘要", "研究目的、核心方法、主要结果与解释边界"],
    ["1 研究问题与总体思路", "为什么PS不能直接当建筑高度；完整处理链"],
    ["2 数据、坐标与高度口径", "输入数据、4 m基底和字段换算"],
    ["3 严格三角面投影与PS表面归属", "距离-多普勒投影、重心坐标和高度比例"],
    ["4 局部强度掩膜精化", "局部阈值、形态学、连通域和几何安全门"],
    ["5 基础异方差稳健平差", "逐PS方程、质量权重、Huber损失和基础结果"],
    ["6 高层建筑顶部恢复", "PS尾部校准、欠估惩罚和78%-90%恢复下限"],
    ["7 两阶段闭环重投影", "重新投影、重新掩膜、重新归属和再平差"],
    ["8 结果与解释", "高层误差、全区高度和典型高楼"],
    ["9 质量控制、局限与使用建议", "QA、科学解释边界和成果使用方式"],
    ["10 输出、复现与后续工作", "文件路径、运行命令和下一步重点"],
  ];
  return table(["章节", "主要内容"], rows, [3300, 6300]);
}

const eAll = HIGH.evaluation.all_highrise;
const eHold = HIGH.evaluation.holdout_fold_1;
const children = [];

// Cover
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 720, after: 220 },
    children: [run("同济校区PS三角面建筑估高", { bold: true, size: 48 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 320 },
    children: [run("掩膜精化与高层顶部恢复技术报告", { bold: true, size: 38 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 360 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 6, color: BLACK, space: 10 },
    },
    children: [run("当前23,178点PS坐标成果 | 4 m统一地面基底", { size: 23 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 900, after: 160 },
    children: [run("324栋有效高度", { bold: true, size: 30 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 160 },
    children: [run("279栋主结果 | 45栋补充级 | 34栋高层顶部恢复", { size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 900, after: 60 },
    children: [run("技术版本：2026-07-26", { size: 21 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [run("同济校区建筑物精细地理编码研究", { size: 18 })],
  }),
  pageBreak(),
);

// Abstract
children.push(
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { after: 220 },
    children: [run("摘要", { bold: true, size: 34 })],
  }),
  para(
    "本报告说明如何利用PS-InSAR散射点、建筑轮廓和严格SAR距离-多普勒投影，估计同济校区建筑高度。方法的核心不是把每个高程点直接落到最近建筑上，而是先把建筑建成由屋顶、墙面和底面组成的三维三角网，再投影到SAR像素坐标，判断PS真正落在哪个建筑表面。随后以三角形重心权重建立高度方程，使用异方差Huber稳健平差估计建筑高度和墙面散射偏差。",
  ),
  para(
    "针对投影三角面内部仍可能包含道路、树木或邻楼亮点的问题，本轮按照参考论文的局部强度思想增加掩膜精化：先在建筑投影外侧估计局部背景，再用亮度阈值、形态学闭运算和连通域筛选收紧屋顶与墙面支持区，而且最终掩膜始终限制在原三角面内部。针对高楼被稳健中心解系统性压低的问题，又增加PS高度第95百分位尾部校准和随建筑高度增加的几何恢复下限。",
  ),
  para(
    `最终共有${HIGH.accepted_buildings}栋建筑保留有效高度，其中${HIGH.primary_buildings}栋为高/中质量主结果，${HIGH.supplementary_buildings}栋为PS支撑补充级，${HIGH.highrise_optimized_buildings}栋高层应用顶部恢复。${eAll.n}栋高层相对既有Shapefile几何先验的MAE由${fmt(eAll.baseline.mae_m)} m降至${fmt(eAll.optimized.mae_m)} m，RMSE由${fmt(eAll.baseline.rmse_m)} m降至${fmt(eAll.optimized.rmse_m)} m；77 m的FID 556由50.04 m恢复到69.30 m。最终重新投影后，从${MAP.candidate_ps_before_mask_refinement.toLocaleString()}个几何候选中保留${MAP.mapped_ps.toLocaleString()}个PS。`,
  ),
  note(
    "重要解释边界",
    "Shapefile高度既用于建筑投影网格，也用于高层恢复下限。因此本文中的高层误差表示模型与既有几何先验的一致性，不是LiDAR、GNSS或水准意义上的独立绝对精度。",
    PALE_ORANGE,
  ),
  h2("关键词"),
  para("PS-InSAR；建筑高度；距离-多普勒投影；三角面；重心坐标；Huber稳健平差；局部强度掩膜；高层顶部恢复", {
    indent: false,
  }),
  pageBreak(),
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { after: 180 },
    children: [run("目录", { bold: true, size: 34 })],
  }),
  contentsTable(),
);

// 1
children.push(
  h1("1 研究问题与总体思路"),
  h2("1.1 为什么建筑高度不能直接从PS高程读取"),
  para(
    "一个PS只代表某个稳定散射中心，它可能来自屋顶、墙角、立面、屋顶设施，甚至来自投影模型范围内的非建筑亮点。同一栋楼的墙面PS通常位于屋顶和地面之间；如果把这些点全部当成屋顶，结果会混乱。反过来，如果只对所有PS取稳健中心，高楼又容易因为低屋顶和墙面点过多而被明显压低。",
  ),
  para(
    "因此，本工作把问题拆成四步：先确定建筑三维表面在SAR图像中的位置；再通过局部幅度收紧有效表面；然后利用PS在三角形内的垂直权重建立高度方程；最后为高层建筑增加顶部恢复分支。这个顺序保证几何、影像证据和高度统计彼此衔接。",
  ),
  h2("1.2 当前处理链"),
  numbered("规范化23,178个PS的SAR像素、平面坐标、高程和质量字段。"),
  numbered("把1,028栋建筑构造成屋顶、墙面和底面三角网，并用严格距离-多普勒模型投影到SAR坐标。"),
  numbered("将投影三角面栅格化，执行局部强度掩膜精化，再把PS分配到屋顶或墙面。"),
  numbered("利用重心坐标建立逐PS高度方程，以异方差Huber IRLS估计建筑高度和墙面偏差。"),
  numbered("对高层建筑使用PS第95百分位尾部证据和高度相关几何下限恢复顶部。"),
  numbered("按优化高度重新投影、重新精化掩膜、重新映射PS并再次平差，形成闭环结果。"),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/06_全区三角面投影.png",
    "图06  全区域建筑屋顶、墙面和底面三角投影",
    { maxWidth: 610, maxHeight: 460 },
  ),
);

// 2
children.push(
  h1("2 数据、坐标与高度口径"),
  h2("2.1 输入数据"),
  table(
    ["数据", "当前内容", "用途"],
    [
      ["建筑", "1,028栋建筑轮廓", "建立三维三角网和几何先验"],
      ["PS点", `${BASE.ps_input.rows.toLocaleString()}点`, "提供SAR位置、平面位置、高程与相干性"],
      ["SAR", "900 × 630像素共注册网格", "三角投影和局部幅度精化"],
      ["地面基底", "4.000 m", "把PS绝对高程转换为建筑离地观测量"],
      ["参考论文", "施展初稿第2.4及3.3-3.7节", "严格投影、表面约束和掩膜思路"],
    ],
    [1800, 2800, 5000],
  ),
  h2("2.2 高度字段为什么容易用错"),
  para(
    "当前PS表同时包含项目内部吴淞高程和相对4 m模型地面的高度。两者表达的是同一物理量在不同基准下的写法。平差代码使用绝对高程字段，同时显式减去4 m基底。",
  ),
  equation("z_i = h_i^(4m) + 4.000", 1),
  para(
    "式中，z_i是第i个PS的项目内部绝对高程，h_i^(4m)是该点高于4 m模型地面的高度。实际进入建筑高度方程的量是z_i - 4，因此与直接使用h_i^(4m)并令基底为0在数学上等价。绝不能对h_i^(4m)再减4 m，否则会重复扣除地面基底。",
  ),
  note(
    "输入追溯",
    `当前PS源文件SHA-256为${BASE.ps_input.sha256}。正式结果必须同时保留输入路径、行数、字段映射和该哈希。`,
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/08_永久散射体高度与建筑平面叠加图.png",
    "图08  PS平面位置、相对4 m地面高度与建筑轮廓叠加",
    { maxWidth: 520, maxHeight: 540 },
  ),
);

// 3
children.push(
  h1("3 严格三角面投影与PS表面归属"),
  h2("3.1 三维建筑模型"),
  para(
    "每栋建筑由底部轮廓和顶部轮廓构成。屋顶和底面被约束三角剖分，轮廓相邻顶点之间形成墙面四边形，每个墙面再拆成两个三角形。与只画一条屋顶线相比，三角面模型能回答PS位于屋顶还是墙面，以及墙面PS处于建筑高度的什么比例。",
  ),
  h2("3.2 距离-多普勒投影"),
  para(
    "对于建筑三维点X，严格投影需要寻找成像时刻t和距离向位置，使点到卫星位置S(t)的斜距与SAR距离一致，同时满足多普勒约束。下面给出概念形式；具体符号正负遵循RSLC参数文件约定。",
  ),
  equation("F_r = ‖X − S(t)‖₂ − R = 0", 2),
  equation("F_d = [X − S(t)] · V(t) − C_D(R,f_D,λ) = 0", 3),
  para(
    "式中V(t)是卫星速度，R是斜距，f_D是多普勒频率，λ是雷达波长，C_D表示由成像参数决定的多普勒项。数值求解得到方位时刻和距离，再转换为方位行、距离列。每个候选高度都必须重新求解这组几何关系，不能用固定像素方向的线性平移代替。",
  ),
  h2("3.3 重心坐标与垂直高度比例"),
  equation("p_i = λ_i1 v_1 + λ_i2 v_2 + λ_i3 v_3,   λ_i1+λ_i2+λ_i3 = 1", 4),
  para(
    "PS像素p_i落入某个三角形后，可以写成三个顶点v_1、v_2、v_3的重心组合。屋顶三角形三个顶点都在顶部，因此高度比例为1；墙面三角形同时含顶部和底部顶点，其垂直比例等于属于顶部顶点的重心权重之和。",
  ),
  equation("f_i = Σ{k∈top} λ_ik", 5),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/04_永久散射体屋顶墙面归属.png",
    "图04  代表建筑的屋顶PS、墙面PS和三角面归属",
    { maxWidth: 540, maxHeight: 500 },
  ),
);

// 4
children.push(
  h1("4 局部强度掩膜精化"),
  h2("4.1 为什么已有三角形还要精化"),
  para(
    "投影三角面给出物理上可能属于建筑的区域，但在SAR影像中，一个三角形内部仍可能穿过暗区、道路亮线、植被或邻楼叠掩。只使用几何包含关系会把这些点错误地分配给建筑。掩膜精化的目标不是移动建筑，而是在不越过投影边界的前提下，找出更符合局部SAR散射的亮区域。",
  ),
  h2("4.2 局部阈值"),
  equation("τ_b = μ_bg,b + 0.35 σ_bg,b", 6),
  para(
    "对每栋建筑b，在初始掩膜外保留1像素间隔，再取5像素宽的背景环。对归一化对数幅度计算局部均值μ_bg,b和标准差σ_bg,b，得到阈值τ_b。屋顶与墙面分别筛选，不使用一个全区固定亮度阈值。",
  ),
  h2("4.3 形态学与几何安全门"),
  equation("M_ref = M_0 ∩ Keep₃{CC[Close₃×₃(A > τ_b)]}", 7),
  para(
    "式中M_0是初始投影掩膜，Close表示一次3×3闭运算，CC表示连通域分解，Keep_3表示每个表面最多保留3个满足面积约束的连通域。最后与M_0相交是最关键的安全门：影像处理只能删减模型内部区域，不能把掩膜扩张到建筑外部。",
  ),
  table(
    ["参数", "当前值", "作用"],
    [
      ["背景间隔", "1 pixel", "避免建筑边缘污染背景"],
      ["背景环宽", "5 pixel", "估计逐建筑局部亮度"],
      ["阈值系数", "0.35 σ", "保留相对背景更亮的散射"],
      ["闭运算", "1次3×3", "连接小间断并填补细小空洞"],
      ["最小连通域", "2 pixel", "去除孤立噪点"],
      ["最多连通域", "3 / 表面", "避免碎片过多"],
      ["空掩膜保护", "最亮3 pixel", "避免整栋建筑完全丢失"],
    ],
    [2800, 2200, 4600],
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/10_局部幅度掩膜精化.png",
    "图10  代表建筑局部掩膜精化：初始模型、强度筛选和最终约束掩膜",
    { maxWidth: 620, maxHeight: 360 },
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/11_全区掩膜精化总览.png",
    "图11  全区域初始掩膜、精化保留区和剔除区；保留FID 826定位框",
    { maxWidth: 620, maxHeight: 330 },
  ),
);

// 5
children.push(
  h1("5 基础异方差稳健平差"),
  h2("5.1 逐PS观测方程"),
  equation("z_i - z_base = f_i H_b + I_wall,i β_b + ε_i", 8),
  para(
    "H_b是建筑b的离地高度；f_i来自三角形重心坐标；I_wall,i在墙面PS处为1、屋顶PS处为0；β_b描述墙面散射相对理想线性高度模型的逐建筑偏差。屋顶PS直接约束H_b，墙面PS通过f_i约束高度，并由β_b吸收共同偏移。",
  ),
  h2("5.2 为什么使用异方差和Huber损失"),
  para(
    "不同PS的可信度并不相同。高相干、位于三角形内部、垂直杠杆充分、建筑重叠歧义小且配准可靠的PS应获得更高权重。少量异常点不能像普通最小二乘那样平方放大，因此采用Huber损失和迭代重加权最小二乘。",
  ),
  equation("min{H_b,β_b}  Σ_i w_i ρ_δ(r_i) + λ_β β_b²", 9),
  equation("ρ_δ(r)=½r²,  ∣r∣≤δ;    ρ_δ(r)=δ(∣r∣−½δ),  ∣r∣>δ", 10),
  para(
    "第一项根据每个PS的基础质量权重w_i和残差稳健权重共同约束高度；第二项把墙面偏差β_b拉向0，当前等效先验尺度为3 m。拟合后还会显式清除异常方程并复算。",
  ),
  h2("5.3 基础结果"),
  table(
    ["指标", "数量或数值"],
    [
      ["输入建筑", "1,028"],
      ["有效投影建筑", "1,005"],
      ["相干性筛选后PS", "20,731"],
      ["掩膜精化后映射PS", BASE.mapped_ps.toLocaleString()],
      ["获得初始高度解", BASE.estimated_buildings_before_final_qc.toLocaleString()],
      ["最终质量控制通过", BASE.final_accepted_buildings.toLocaleString()],
      ["高/中质量主结果", BASE.final_primary_buildings.toLocaleString()],
      ["PS支撑补充级", BASE.final_supplementary_buildings.toLocaleString()],
      ["高度中位数", `${fmt(BASE.height_m.median, 2)} m`],
      ["高度均值", `${fmt(BASE.height_m.mean, 2)} m`],
      ["高度范围", `${fmt(BASE.height_m.min, 2)}-${fmt(BASE.height_m.max, 2)} m`],
    ],
    [5000, 4600],
  ),
  h3("5.4 补充级如何增加覆盖数量"),
  para(
    "补充级不是用Shapefile高度、均值或邻域高度填空，而是从已有有限平差解中保留证据稍弱但仍可审计的建筑。普通建筑至少需要2条有效PS方程、1个屋顶PS、1个A/B级强PS，有效PS数不低于1.5，不确定度不超过12 m，并继续限制残差、内点率、墙面偏差和屋顶-墙面差异。先验高度不低于30 m的高层补充级额外要求至少3条方程，使其能够进入顶部恢复链，避免把只有2条方程且明显偏低的高楼列为正式结果。",
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/07_全区永久散射体质量评估.png",
    "图07  全区域PS质量等级与建筑支持情况",
    { maxWidth: 610, maxHeight: 460 },
  ),
  note(
    "基础模型的局限",
    "Huber中心解能抵抗异常值，但对高楼而言，较低屋顶、墙面和设施散射点可能占多数。此时“稳健”会变成向中部收缩，不能把保守低估误认为可靠。",
    PALE_ORANGE,
  ),
);

// 6
children.push(
  h1("6 高层建筑顶部恢复"),
  h2("6.1 从中心估计转向顶部证据"),
  para(
    "高层分支只处理几何先验不低于30 m、基础结果通过质量控制且至少有3个可靠PS的建筑。逐建筑计算PS校正高度的第50、75、95和99百分位。第95百分位用于模型特征，第99百分位仅用于审计，因为单个极端PS可能把第99百分位或最大值推得过高。",
  ),
  h2("6.2 非对称Huber-ridge校准"),
  equation("H_cal = β_0 + β_1 x_(q95,std) + β_2 x_(center,std)", 11),
  para(
    `星号表示标准化特征。x_q95是可靠PS逐点校正高度的第95百分位，x_center是基础稳健中心。模型用200 m棋盘格第0折的${HIGH.parameters.calibration.training_buildings}栋高层拟合，ridge系数为${HIGH.parameters.ridge}，Huber阈值为${HIGH.parameters.calibration.huber_c}。`,
  ),
  equation("w_asym(r) = 1.25  (H_ref-H_cal>0);   1  (otherwise)", 12),
  para(
    "当H_ref-H_cal>0时，模型正在欠估，因此把该残差权重提高到1.25。这样做不会直接复制先验高度，但会让模型在同等误差下更重视欠估。",
  ),
  h2("6.3 防止最高端再次被压缩"),
  para(
    "只有非对称回归仍可能把77 m或更高建筑压向总体中心。为此增加一个随先验高度平滑增加的恢复下限。它在30 m处为先验的78%，随后每增加1 m提高0.3个百分点，到70 m及以上封顶为90%。",
  ),
  equation("r_floor(H_ref)=min[0.90, 0.78+0.003·max(H_ref-30,0)]", 13),
  equation("H_floor = r_floor(H_ref) H_ref", 14),
  equation("H_upper = min(1.5H_ref, H_ref+25)", 15),
  equation("H_candidate=min[max(H_cal,H_floor),H_upper]", 16),
  equation("H_optimized=max(H_baseline,H_candidate)", 17),
  para(
    "最终规则不会降低已经较高的基础解。候选至少比基础高度高1 m才应用。FID 725的基础高度79.86 m已经接近83 m先验，因此保持不变；FID 556的基础高度50.04 m远低于77 m先验，恢复下限将其提高到69.30 m。",
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/highrise_optimized/png/09_高层优化验证.png",
    `高层图09  ${eAll.n}栋高层优化前后与几何先验的内部一致性对比`,
    { maxWidth: 610, maxHeight: 440 },
  ),
);

// 7
children.push(
  h1("7 两阶段闭环重投影"),
  h2("7.1 为什么不能只改最终CSV"),
  para(
    "建筑高度改变后，屋顶和墙面在SAR图像中的位置会变化，PS对应的建筑、表面类别、重心权重和局部掩膜也可能变化。如果只修改高度表而继续使用旧PS归属，几何与平差将不一致。因此高层优化必须重新进入投影和映射链。",
  ),
  numbered("基础结果上计算第一次顶部恢复高度。", "report-closed-loop"),
  numbered("用第一次高度重投影全部建筑三角面。", "report-closed-loop"),
  numbered("重新计算局部背景、强度阈值、连通域和PS表面归属。", "report-closed-loop"),
  numbered("重新生成PS高度方程并执行异方差Huber墙面偏差平差。", "report-closed-loop"),
  numbered("在新平差结果上再次执行高层顶部恢复。", "report-closed-loop"),
  numbered("用最终高度生成final_mapping和全部正式图件。", "report-closed-loop"),
  h2("7.2 最终映射变化"),
  table(
    ["指标", "基础映射", "高层最终映射"],
    [
      ["几何候选PS", BASE.candidate_ps_before_mask_refinement.toLocaleString(), MAP.candidate_ps_before_mask_refinement.toLocaleString()],
      ["精化后PS", BASE.mapped_ps.toLocaleString(), MAP.mapped_ps.toLocaleString()],
      ["屋顶PS", BASE.surface_counts.roof.toLocaleString(), MAP.surface_counts.roof.toLocaleString()],
      ["墙面PS", BASE.surface_counts.wall.toLocaleString(), MAP.surface_counts.wall.toLocaleString()],
      ["触发空掩膜保护建筑", BASE.mask_refinement.fallback_buildings.toLocaleString(), MAP.mask_refinement.fallback_buildings.toLocaleString()],
    ],
    [3400, 3100, 3100],
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/highrise_optimized/png/10_三角面投影与永久散射体融合.png",
    "高层图10  最终全区域三角面与PS融合；右侧保留FID 569局部放大",
    { maxWidth: 620, maxHeight: 390 },
  ),
);

// 8
children.push(
  h1("8 结果与解释"),
  h2("8.1 高层结果"),
  equation("Bias=(1/n)Σ_i(Ĥ_i-H_i)", 18),
  equation("MAE=(1/n)Σ_i abs(Ĥ_i−H_i)", 19),
  equation("RMSE=sqrt[(1/n)Σ_i(Ĥ_i−H_i)²]", 20),
  table(
    ["指标", "基础中心解", "顶部恢复"],
    [
      ["高层样本", `${eAll.n}`, `${eAll.n}`],
      ["平均偏差", `${fmt(eAll.baseline.bias_m)} m`, `+${fmt(eAll.optimized.bias_m)} m`],
      ["MAE", `${fmt(eAll.baseline.mae_m)} m`, `${fmt(eAll.optimized.mae_m)} m`],
      ["RMSE", `${fmt(eAll.baseline.rmse_m)} m`, `${fmt(eAll.optimized.rmse_m)} m`],
      ["估计/先验中位比", fmt(eAll.baseline.median_ratio_to_reference, 3), fmt(eAll.optimized.median_ratio_to_reference, 3)],
      ["最大欠估", "54.22 m", "7.85 m"],
    ],
    [3400, 3100, 3100],
  ),
  para(
    `未参与校准系数拟合的空间第1折共有${eHold.n}栋高层，基础MAE为${fmt(eHold.baseline.mae_m)} m，顶部恢复后为${fmt(eHold.optimized.mae_m)} m；RMSE由${fmt(eHold.baseline.rmse_m)} m降至${fmt(eHold.optimized.rmse_m)} m。由于每栋建筑的恢复下限使用了Shapefile先验，而且模型开发阶段已经查看空间诊断结果，这一组只能称为空间诊断，不能称为完全独立验证。`,
  ),
  h2("8.2 全部建筑结果"),
  table(
    ["指标", "最终值"],
    [
      ["保留有效高度", `${HIGH.accepted_buildings}栋`],
      ["高层顶部恢复", `${HIGH.highrise_optimized_buildings}栋`],
      ["高度最小值", `${fmt(HIGH.optimized_height_m.minimum)} m`],
      ["高度中位数", `${fmt(HIGH.optimized_height_m.median)} m`],
      ["高度均值", `${fmt(HIGH.optimized_height_m.mean)} m`],
      ["高度最大值", `${fmt(HIGH.optimized_height_m.maximum)} m`],
      ["最终映射PS", MAP.mapped_ps.toLocaleString()],
    ],
    [5000, 4600],
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/09_高层优化建筑高度估计.png",
    "正式图09  全区域最终建筑高度估计；图05已取消，本图为正式高度主图",
    { maxWidth: 535, maxHeight: 555 },
  ),
);

// 9
children.push(
  h1("9 质量控制、局限与使用建议"),
  h2("9.1 已完成的质量检查"),
  bullet(`最终建筑表为1,028行，fid唯一；${HIGH.accepted_buildings}栋具有最终高度，其中${HIGH.primary_buildings}栋为主结果、${HIGH.supplementary_buildings}栋为补充级。`),
  bullet("最终建筑GeoPackage包含1,028个有效几何，坐标系为EPSG:4326。"),
  bullet("最终雷达三角面为19,896个有效几何，坐标语义为SAR像素。"),
  bullet(`最终PS矢量为${MAP.mapped_ps.toLocaleString()}个有效点，坐标系为EPSG:4326。`),
  bullet("掩膜精化结果始终是初始投影模型的子集。"),
  bullet("14项几何、配准、掩膜和平差单元测试全部通过。"),
  bullet("高层PNG可解码，SVG通过XML解析并保留可编辑文字。"),
  bullet("当前结果目录不生成PDF图件；本报告PDF单独存放在output/pdf。"),
  h2("9.2 主要局限"),
  bullet("PS高程采用项目内部地面参考的吴淞高程口径，尚无独立LiDAR、GNSS或水准验证。"),
  bullet("所有建筑使用统一4 m基底，未考虑逐建筑地面起伏和局部台阶。"),
  bullet("Shapefile高度参与投影几何和高层恢复下限，先验一致性不能替代真实精度。"),
  bullet("SAR叠掩、阴影和同一亮散射体被多栋建筑竞争仍可能造成归属歧义。"),
  bullet(`高层样本只有${eAll.n}栋，不能把当前校准系数直接推广到不同城市、传感器或成像几何。`),
  h2("9.3 推荐使用方式"),
  para(
    "需要全区建筑高度时使用height_optimized_m，并同时保留final_quality、PS数量和是否触发高层恢复。分析普通建筑可优先采用高、中质量基础解；分析高层时必须同时查看PS尾部分位、恢复下限和几何先验，不能只引用单一高度值。",
  ),
  note(
    "对外表述建议",
    `推荐写作“PS三角面模型给出的建筑高度估计”或“与Shapefile投影几何一致的高层顶部恢复结果”。在获得独立控制前，不应写作“真实建筑高度精度为${fmt(eAll.optimized.mae_m)} m”。`,
    PALE_ORANGE,
  ),
);

// 10
children.push(
  h1("10 输出、复现与后续工作"),
  h2("10.1 关键输出"),
  table(
    ["内容", "相对路径"],
    [
      ["最终逐建筑CSV", "highrise_optimized/tables/building_height_estimates_highrise_optimized.csv"],
      ["最终建筑矢量", "highrise_optimized/vectors/building_height_estimates_highrise_optimized.gpkg"],
      ["最终PS映射", "highrise_optimized/final_mapping/vectors/ps_points_on_building_surfaces.gpkg"],
      ["最终雷达三角面", "highrise_optimized/final_mapping/triangles/building_surface_triangles_radar.gpkg"],
      ["方法和指标摘要", "highrise_optimized/summary.json"],
      ["高层验证图", "highrise_optimized/png/09_高层优化验证.png"],
      ["全区与局部融合图", "highrise_optimized/png/10_三角面投影与永久散射体融合.png"],
      ["正式全区域高度图", "png/09_高层优化建筑高度估计.png"],
    ],
    [2800, 6800],
  ),
  h2("10.2 复现命令"),
  para("在项目根目录运行：", { indent: false }),
  new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: [CONTENT_WIDTH],
    rows: [
      new TableRow({
        children: [
          new TableCell({
            width: { size: CONTENT_WIDTH, type: WidthType.DXA },
            borders: { top: thick, bottom: thick, left: none, right: none },
            margins: { top: 160, bottom: 160, left: 180, right: 180 },
            children: [
              new Paragraph({
                spacing: { after: 80 },
                children: [run("MPLCONFIGDIR=/tmp/matplotlib-highrise-top-restoration \\", { size: 17 })],
              }),
              new Paragraph({
                spacing: { after: 80 },
                children: [run("/home/u/geocoding/tongji_sbas/.venv/bin/python \\", { size: 17 })],
              }),
              new Paragraph({
                spacing: { after: 80 },
                children: [run("  ps_triangle_height_estimation/code/run_highrise_envelope_optimization.py \\", { size: 17 })],
              }),
              new Paragraph({
                spacing: { after: 80 },
                children: [run("  --source-root ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current \\", { size: 17 })],
              }),
              new Paragraph({
                children: [run("  --output-root ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/highrise_optimized", { size: 17 })],
              }),
            ],
          }),
        ],
      }),
    ],
  }),
  h2("10.3 后续优先级"),
  numbered("引入逐建筑地面高程，替代统一4 m基底。", "report-future"),
  numbered("获取覆盖不同高度和建筑形态的独立LiDAR或测量控制。", "report-future"),
  numbered("对最高建筑增加多时相、多入射角或多轨SAR约束，减少单一几何下的顶部缺失。", "report-future"),
  numbered("把建筑间PS竞争升级为全区域联合归属，而不是逐建筑局部决策。", "report-future"),
  numbered("按高度层级、建筑形态和PS可观测性分别报告覆盖率与误差。", "report-future"),
  para(
    "参考：施展《附加轮廓矢量的SAR建筑物精细地理编码（初稿）》；当前工作与结果摘要见agent.md和highrise_optimized/summary.json。",
    { indent: false, before: 120, after: 0 },
  ),
);

// Concise report: focus on projection and building-height estimation.
// The longer chapter set above is retained as source material, while only this
// streamlined sequence is emitted to the current report.
const conciseChildren = [];

conciseChildren.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 900, after: 240 },
    children: [run("同济校区PS三角面建筑估高", { bold: true, size: 48 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
    children: [run("投影与高度估计方法原理", { bold: true, size: 38 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 540 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 6, color: BLACK, space: 10 },
    },
    children: [run("通俗详解版技术报告", { size: 23 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 720, after: 150 },
    children: [run("324栋有效高度", { bold: true, size: 30 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 150 },
    children: [run("279栋主结果  |  45栋补充级  |  34栋高层顶部恢复", { size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 900 },
    children: [run("技术版本：2026-07-27", { size: 20 })],
  }),
  pageBreak(),
);

conciseChildren.push(
  h1("先说清楚：PS三角面法到底怎样估高"),
  para(
    "这套方法不是把一栋楼附近的PS高程直接求平均。原因很简单：PS可能落在屋顶，也可能落在墙面中部；一个18 m高的墙面PS，并不说明整栋楼只有18 m高。方法先判断每个PS属于哪栋建筑、落在屋顶还是墙面，以及它处于楼身高度的什么比例，再把许多PS写成方程共同求建筑高度。",
  ),
  note(
    "一句话理解",
    "先用三维投影给每个PS找到正确的“楼”和“楼身位置”，再把屋顶PS和墙面PS当成多把刻度不同的高度尺，用稳健平差合成整栋建筑高度。",
  ),
  h2("一个最小数值例子"),
  table(
    ["PS类型", "离地观测", "表面比例f", "它真正提供的约束"],
    [
      ["屋顶PS", "39 m", "1.0", "39≈1.0×建筑高度"],
      ["墙面PS", "18 m", "0.6", "18≈0.6×建筑高度+墙面偏差"],
      ["墙面PS", "10 m", "0.3", "10≈0.3×建筑高度+墙面偏差"],
    ],
    [1900, 1800, 1800, 4100],
  ),
  para(
    "三条观测都指向一栋约30至40 m的建筑，而不是把39、18、10直接平均成22.3 m。实际处理会使用更多PS，并按相干性、表面归属和残差给出不同权重。",
  ),
  h2("完整处理链"),
  table(
    ["核心问题", "采用的方法", "得到的结果"],
    [
      ["建筑在SAR图上的什么位置？", "三维三角面与严格距离-多普勒投影", "屋顶、墙面和底面的SAR像素范围"],
      ["一个PS属于屋顶还是墙面？", "三角形包含关系、重心坐标和局部强度掩膜", "建筑编号、表面类型和垂直比例f(i)"],
      ["许多PS怎样合成一栋楼的高度？", "逐PS观测方程、质量权重和Huber稳健平差", "基础建筑高度H(b)及其质量等级"],
      ["为什么高楼不能只取稳健中心？", "PS第95百分位、非对称校准和高度恢复下限", "不过度保守的高层顶部高度"],
      ["高度改变后几何怎么办？", "按新高度重新投影、重新掩膜和重新平差", "几何位置与高度相互一致的闭环结果"],
    ],
    [3100, 3600, 2900],
  ),
  h2("用一栋建筑理解全过程"),
  para(
    "先把建筑轮廓向上抬成三维屋顶和墙面，并投影到SAR图上；再查看哪些PS真正落在这些表面内。屋顶PS直接“量到楼顶”，墙面PS只量到楼身某个比例位置。平差把这些质量不同的尺子合在一起。若建筑很高而顶部PS稀少，再读取可靠PS分布的高端尾部，避免把真实高楼压成普通楼。最后用新高度重新投影检查，确保屋顶位置也随高度正确移动。",
  ),
  note(
    "最核心的逻辑",
    "投影决定观测有没有用对，平差决定多个观测怎样合并，高层恢复决定顶部证据不足时怎样避免系统性低估。三者缺一不可。",
  ),
);

conciseChildren.push(
  h1("摘要"),
  para(
    "本报告重点回答两个问题：第一，怎样把地图中的建筑物准确投影到SAR影像；第二，怎样利用落在建筑表面的PS散射点估计建筑高度。核心思路是先把建筑表示成屋顶和墙面的三角网，通过严格距离-多普勒模型投影到SAR像素，再用局部影像强度收紧投影掩膜；随后根据PS在三角形中的位置建立逐点高度方程，用稳健平差求建筑高度，并为高楼增加顶部恢复，避免结果因求稳而系统性偏低。",
  ),
  para(
    `当前23,178个PS中，最终有${MAP.mapped_ps.toLocaleString()}个获得建筑表面归属，共得到${HIGH.accepted_buildings}栋有效建筑高度。${eAll.n}栋高层相对现有Shapefile几何先验的MAE由${fmt(eAll.baseline.mae_m)} m降至${fmt(eAll.optimized.mae_m)} m，RMSE由${fmt(eAll.baseline.rmse_m)} m降至${fmt(eAll.optimized.rmse_m)} m。`,
  ),
  note(
    "一句话理解",
    "投影解决“这个PS属于哪栋楼、落在屋顶还是墙面”；高度估计解决“这个PS对整栋楼高度贡献多少”。两者必须闭环，不能只在最后一张高度表里修改数值。",
  ),
  h2("报告结构"),
  table(
    ["部分", "重点"],
    [
      ["1 投影方法", "三维三角面、距离-多普勒投影、PS表面归属、掩膜精化"],
      ["2 高度估计", "逐PS观测方程、稳健平差、高楼顶部恢复、闭环重投影"],
      ["3 结果与边界", "全区成果、精度变化、适用范围与局限"],
    ],
    [2500, 7100],
  ),
  pageBreak(),
);

conciseChildren.push(
  h1("1 投影方法：把建筑放到正确的SAR位置"),
  h2("1.1 先把建筑变成可计算的三维表面"),
  para(
    "普通建筑轮廓只有平面边界，无法说明雷达照射后屋顶和墙面分别出现在哪里。因此，先复制一份轮廓作为底部，再按建筑高度抬升得到顶部；屋顶被剖分成三角形，相邻的上下轮廓边组成墙面，每个墙面也拆成两个三角形。这样，一栋建筑就不再是一条线，而是由许多小三角面组成的三维外壳。",
  ),
  para(
    "三角面很适合SAR投影：三个顶点确定一个平面，投影后仍是一个三角形；PS落入其中后，还可以用三个顶点的权重描述它在表面上的位置。",
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/06_全区三角面投影.png",
    "图1  全区域建筑屋顶、墙面和底面三角投影",
    { maxWidth: 610, maxHeight: 460 },
  ),
  h2("1.2 严格距离-多普勒投影"),
  para(
    "SAR影像不是从正上方拍摄的平面照片。一个三维点在影像中的位置，同时取决于它到卫星的距离以及卫星沿轨运动产生的多普勒条件。对每个建筑顶点，需要寻找一个成像时刻t和斜距R，使下面两个条件同时成立。",
  ),
  equation("距离条件：‖X − S(t)‖₂ − R = 0", 1),
  equation("多普勒条件：[X − S(t)] · V(t) − C_D(R, f_D, λ) = 0", 2),
  para(
    "X是建筑三维点，S(t)和V(t)分别是卫星位置与速度。第一式保证距离正确，第二式保证多普勒条件正确。求解后，把t和R换算成方位行与距离列，就得到该点的SAR像素坐标。建筑高度一旦改变，顶点位置、斜距和多普勒关系也会改变，所以必须重新求解，不能简单把屋顶沿固定方向平移几个像素。",
  ),
);

conciseChildren.push(
  h2("1.3 判断PS落在屋顶还是墙面"),
  para(
    "投影后三角形覆盖SAR像素。若PS点位于某个三角形内，它可表示为三个投影顶点的加权平均；这三个权重称为重心坐标。",
  ),
  equation("p(i) = λ(i,1)·v(1) + λ(i,2)·v(2) + λ(i,3)·v(3)，Σλ(i,k)=1", 3),
  para(
    "屋顶三角形的三个顶点都在建筑顶部，因此PS对应的垂直比例为1。墙面三角形既有顶部顶点，也有底部顶点；把顶部顶点的重心权重相加，就得到该PS位于整栋楼高度的比例f_i。",
  ),
  equation("f(i) = Σ［顶部顶点 k］λ(i,k)", 4),
  para(
    "例如，f_i=1表示点在屋顶；f_i=0.6表示它在墙面约60%的高度处。由此，墙面PS不再被粗暴地当作屋顶点，而是按其实际垂直位置参与估高。",
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/04_永久散射体屋顶墙面归属.png",
    "图2  PS与屋顶、墙面三角形的归属关系",
    { maxWidth: 540, maxHeight: 500 },
  ),
  pageBreak(),
  h2("1.4 用局部SAR强度精化掩膜"),
  para(
    "几何投影只说明“可能属于建筑”的范围，但这个范围内仍可能混入道路、树木或邻楼亮点。为此，在每栋建筑周围单独估计背景亮度，不使用一个全区统一阈值。当前阈值为局部背景均值加0.35倍标准差。",
  ),
  equation("τ(b) = μ(bg,b) + 0.35·σ(bg,b)", 5),
  para(
    "超过阈值的区域经过一次3×3闭运算，以连接小间断；再删除过小碎片，并限制每个表面最多保留3个连通区域。最终结果始终与原始投影掩膜相交。",
  ),
  equation("M(ref) = M(0) ∩ Keep₃{CC[Close₃×₃(A > τ(b))]}", 6),
  note(
    "最重要的安全门",
    "强度处理只能在投影三角形内部删减可疑区域，不能把掩膜扩张到建筑外。这样既利用SAR亮度，又不破坏几何约束。",
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/10_局部幅度掩膜精化.png",
    "图3  局部掩膜精化：几何候选、强度筛选与最终保留区",
    { maxWidth: 620, maxHeight: 360 },
  ),
);

conciseChildren.push(
  h1("2 建筑高度估计：把每个PS变成一条高度约束"),
  h2("2.1 高程基准"),
  para(
    "当前建筑高度以统一4.000 m模型地面为零点。PS表中的绝对高程z_i先减去4 m，得到相对地面的观测量。若直接使用表内已经相对4 m地面的高度字段，就不能再次减4 m。",
  ),
  equation("h(obs,i) = z(i) − 4.000", 7),
  h2("2.2 逐PS高度方程"),
  equation("z(i) − z(base) = f(i)·H(b) + I(wall,i)·β(b) + ε(i)", 8),
  para(
    "H_b是待求的建筑高度，f_i来自投影三角形的重心坐标。屋顶PS的f_i=1，直接约束整栋楼高度；墙面PS的f_i介于0和1，只约束相应比例。I_wall,i在墙面点上取1，β_b用于吸收立面散射相对理想线性模型的共同偏移，ε_i表示随机误差。",
  ),
  para(
    "通俗地说：屋顶点像一把直接量到楼顶的尺子；墙面点只量到楼身某个比例位置。投影步骤给出了这把“比例尺”，平差步骤再把许多长短不同、质量不同的尺子合在一起。",
  ),
  h2("2.3 异方差Huber稳健平差"),
  para(
    "不同PS的可靠程度不同。相干性高、位于三角形内部、归属歧义小的点权重更大。少量异常点若使用普通最小二乘会被平方放大，因此采用Huber损失：小残差按平方处理，大残差改为近似线性增长。",
  ),
  equation("对 H(b)、β(b) 求最小：Σ w(i)·ρδ[r(i)] + λβ·β(b)²", 9),
  equation("ρδ(r)=½r²（|r|≤δ）；ρδ(r)=δ(|r|−½δ)（|r|>δ）", 10),
  para(
    "通过迭代重加权求解后，再清除明显异常方程并复算。它能稳定普通建筑结果，但对高楼存在一个特殊问题：墙面点和较低屋面点往往多于真正楼顶点，稳健中心可能把高度压向楼身中部。",
  ),
  equation("w(Huber,i)=1（|r(i)|≤δ）；δ/|r(i)|（|r(i)|>δ）", 11),
  para(
    "这条权重公式说明Huber为什么既稳又不会完全丢弃数据：残差正常的PS保留完整权重；残差越大，权重按δ/|r|逐渐减小，而不是像普通最小二乘那样让异常点以残差平方支配结果。",
  ),
  note(
    "一个墙面PS的例子",
    "若PS绝对高程为22 m，地面基底为4 m，则观测离地高度为18 m。若投影表明它位于墙面60%处，则方程近似为18 = 0.6×建筑高度 + 墙面共同偏差。它不能直接被当成18 m屋顶。",
  ),
);

conciseChildren.push(
  pageBreak(),
  h2("2.4 高楼不能因为求稳而保守：顶部恢复"),
  para(
    "对几何先验不低于30 m、基础平差通过质量控制且至少有3个可靠PS的建筑，额外读取PS高度分布的上尾信息。第95百分位代表稳定的高端证据，比最大值更不容易被单个异常点抬高；同时保留基础稳健中心，组成非对称Huber-ridge校准。",
  ),
  equation("H(cal) = β₀ + β₁·x(q95,std) + β₂·x(center,std)", 12),
  para(
    "当模型低估时，残差权重提高到1.25，使训练更重视欠估。为了避免最高建筑仍被回归到总体平均值，再设置随先验高度增加的恢复下限：30 m建筑至少恢复到先验的78%，比例随高度增加，并在70 m以上封顶为90%。",
  ),
  equation("r(floor)=min[0.90，0.78+0.003·max(H(ref)−30，0)]", 13),
  equation("H(candidate)=min{max[H(cal)，r(floor)·H(ref)]，min[1.5H(ref)，H(ref)+25]}", 14),
  equation("H(optimized)=max[H(baseline)，H(candidate)]", 15),
  para(
    "最后一式保证顶部恢复只会抬升明显偏低的高楼，不会把已有较高且稳定的基础结果向下拉。以FID 556为例，基础结果为50.04 m，几何先验为77 m，最终恢复到69.30 m；FID 725的基础结果79.86 m已接近83 m先验，因此保持不变。",
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/highrise_optimized/png/09_高层优化验证.png",
    "图4  高层建筑顶部恢复前后对比",
    { maxWidth: 610, maxHeight: 400 },
  ),
  pageBreak(),
  h2("2.5 高度改变后必须闭环重投影"),
  para(
    "高楼被抬高后，它的屋顶和墙面在SAR影像中的位置也随之改变。因此，最终流程不是简单改一列高度，而是按新高度重新构建三角面、重新执行距离-多普勒投影、重新精化掩膜、重新分配PS并再次平差，直到几何位置与高度结果一致。",
  ),
  table(
    ["闭环步骤", "会随高度变化的量"],
    [
      ["1 重建三维三角面", "屋顶顶点高程和墙面几何"],
      ["2 重新严格投影", "屋顶、墙面在SAR中的行列位置"],
      ["3 重新精化掩膜", "局部背景、亮区和连通区域"],
      ["4 重新分配PS", "建筑编号、屋顶/墙面类型和重心比例"],
      ["5 重新执行平差", "逐PS方程、残差、权重和建筑高度"],
    ],
    [3200, 6400],
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/highrise_optimized/png/10_三角面投影与永久散射体融合.png",
    "图5  最终三角面与PS融合，右侧保留典型建筑局部放大",
    { maxWidth: 620, maxHeight: 350 },
  ),
);

conciseChildren.push(
  h1("3 当前结果与解释边界"),
  h2("3.1 结果概览"),
  table(
    ["指标", "当前结果"],
    [
      ["有效建筑高度", `${HIGH.accepted_buildings}栋`],
      ["高/中质量主结果", `${HIGH.primary_buildings}栋`],
      ["PS支持补充级", `${HIGH.supplementary_buildings}栋`],
      ["高层顶部恢复", `${HIGH.highrise_optimized_buildings}栋`],
      ["最终获得表面归属的PS", MAP.mapped_ps.toLocaleString()],
      ["高层MAE", `${fmt(eAll.baseline.mae_m)} m → ${fmt(eAll.optimized.mae_m)} m`],
      ["高层RMSE", `${fmt(eAll.baseline.rmse_m)} m → ${fmt(eAll.optimized.rmse_m)} m`],
      ["最大欠估", "54.22 m → 7.85 m"],
    ],
    [5200, 4400],
  ),
  ...figure(
    "ps_triangle_height_estimation/results/picall/touying2_ps_coordinates_current/png/09_高层优化建筑高度估计.png",
    "图6  全区域最终建筑高度估计（原图05取消，以结果目录图09为正式主图）",
    { maxWidth: 535, maxHeight: 555 },
  ),
  h2("3.2 怎样正确理解这些数字"),
  para(
    "Shapefile高度既参与初始建筑投影，也参与高层恢复下限，因此上表的高层误差表示结果与现有几何先验的一致程度，不等同于独立真实精度。当前结果适合用于比较建筑间高度、检查投影与PS归属、识别明显低估的高楼；若要对外声明绝对高度精度，仍需LiDAR、GNSS或水准控制。",
  ),
  table(
    ["当前优势", "仍需改进"],
    [
      ["严格投影把PS归属到具体屋顶或墙面", "统一4 m地面应升级为逐建筑地面高程"],
      ["局部强度掩膜减少道路、植被和邻楼干扰", "叠掩和多建筑竞争仍可能造成歧义"],
      ["高楼顶部恢复显著减少保守低估", "需要独立高度控制验证恢复系数"],
      ["高度优化后重新投影，几何与结果闭环", "可增加多轨、多入射角或多时相约束"],
    ],
    [4800, 4800],
  ),
  note(
    "结论",
    "当前方法的关键进步不是单独换一个估高公式，而是把三维投影、PS表面归属、局部掩膜、稳健平差和高楼顶部恢复连成闭环。投影决定观测是否用对，高层分支则避免稳健模型把真正的高楼估得过于保守。",
  ),
  para(
    "参考：施展《附加轮廓矢量的SAR建筑物精细地理编码（初稿）》；详细参数与结果表见highrise_optimized/summary.json。",
    { indent: false, before: 140, after: 0 },
  ),
);

const doc = new Document({
  creator: "Codex",
  title: "同济校区PS三角面建筑估高方法原理报告",
  description: "通俗说明三角面投影、掩膜精化、稳健估高、高层顶部恢复和闭环重投影",
  styles: {
    default: {
      document: {
        run: { font: FONT, size: 21, color: BLACK },
        paragraph: {},
      },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: BLACK },
        paragraph: { spacing: { before: 260, after: 220 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: BLACK },
        paragraph: { spacing: { before: 220, after: 140 }, outlineLevel: 1 },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: FONT, size: 23, bold: true, color: BLACK },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "report-bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 600, hanging: 300 } } },
          },
          {
            level: 1,
            format: LevelFormat.BULLET,
            text: "–",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1000, hanging: 300 } } },
          },
        ],
      },
      {
        reference: "report-steps",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 650, hanging: 350 } } },
          },
        ],
      },
      {
        reference: "report-closed-loop",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 650, hanging: 350 } } },
          },
        ],
      },
      {
        reference: "report-future",
        levels: [
          {
            level: 0,
            format: LevelFormat.DECIMAL,
            text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 650, hanging: 350 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH, height: PAGE_HEIGHT },
          margin: { top: 1080, right: 1153, bottom: 1080, left: 1153 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              border: {
                bottom: { style: BorderStyle.SINGLE, size: 4, color: BLACK, space: 4 },
              },
              children: [
                run("同济校区PS三角面建筑估高方法原理报告", {
                  size: 16,
                  color: GRAY,
                }),
              ],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              border: {
                top: { style: BorderStyle.SINGLE, size: 4, color: BLACK, space: 4 },
              },
              children: [
                run("内部技术报告  |  第 ", { size: 16, color: GRAY }),
                new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: GRAY }),
                run(" 页", { size: 16, color: GRAY }),
              ],
            }),
          ],
        }),
      },
      children: conciseChildren,
    },
  ],
});

fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUTPUT, buffer);
  process.stdout.write(`${OUTPUT}\n`);
});
