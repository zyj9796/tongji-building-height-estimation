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
  Packer,
  PageBreak,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  VerticalAlign,
  WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const WORKSPACE_ROOT = path.resolve(ROOT, "../..");
const PICALL = path.join(ROOT, "results", "PICALL");
const SUMMARY = JSON.parse(
  fs.readFileSync(
    path.join(ROOT, "results", "tables", "joint_quantity_quality_optimization_summary.json"),
    "utf8",
  ),
);
const ASSET_DIR = path.join(WORKSPACE_ROOT, "tmp", "pixel_offset_picall_report_assets");
const OUTPUT = path.join(
  WORKSPACE_ROOT,
  "output",
  "像素偏移建筑高度估计PICALL结果",
  "像素偏移建筑高度估计方法原理报告.docx",
);

const PAGE_WIDTH = 11906;
const PAGE_HEIGHT = 16838;
const CONTENT_WIDTH = 9600;
const FONT = "Noto Sans CJK SC";
const BLACK = "000000";
const none = { style: BorderStyle.NONE, size: 0, color: BLACK };
const thin = { style: BorderStyle.SINGLE, size: 4, color: BLACK };
const thick = { style: BorderStyle.SINGLE, size: 8, color: BLACK };

const FIGURES = [
  {
    file: "02_先验投影与合成孔径雷达校正.svg",
    title: "先验投影与SAR局部改正",
    group: "方法诊断",
    note: "对比建筑先验投影与局部SAR特征改正后的屋顶位置，说明像素偏移如何改变投影位置。",
  },
  {
    file: "03_像素偏移质量审计.svg",
    title: "基础像素偏移质量审计",
    group: "质量诊断",
    note: "汇总基础分支的偏移、评分和质量分布，用于识别边界解、弱匹配和不稳定建筑。",
  },
  {
    file: "04_像素偏移单体建筑诊断.svg",
    title: "代表建筑单体诊断",
    group: "单体诊断",
    note: "用单栋建筑展示投影、局部匹配和高度搜索过程，帮助解释单体结果的来源。",
  },
  {
    file: "05_全部建筑零米高程投影.svg",
    title: "全部建筑0 m绝对高程投影",
    group: "几何基准",
    note: "全部建筑按0 m绝对高程执行严格距离-多普勒顶面投影，是估计每米高度像素位移的几何基准之一。",
  },
  {
    file: "06_全部建筑矢量高度投影.svg",
    title: "全部建筑按Shapefile高度投影",
    group: "几何基准",
    note: "全部建筑按Shapefile height字段作为屋顶绝对高程重新投影，与0 m投影共同建立高度与像素位置的关系。",
  },
  {
    file: "07_矢量高度局部雷达校正.svg",
    title: "Shapefile高度起点的局部SAR改正",
    group: "中间结果",
    note: "从先验高度投影位置出发，依据局部SAR特征搜索更合理的屋顶位置，是早期局部改正分支。",
  },
  {
    file: "08_合成孔径雷达建筑特征增强.svg",
    title: "SAR建筑特征增强",
    group: "影像处理",
    note: "展示三景RSLC经对数压缩、保边去斑、局部对比和边缘融合后的建筑特征；增强影像只用于匹配评分。",
  },
  {
    file: "09_形态自适应增强雷达校正.svg",
    title: "形态自适应增强配准",
    group: "中间结果",
    note: "根据建筑面积、长宽比和尺度调整搜索窗口与评分参数，提高不同建筑形态下的局部匹配适应性。",
  },
  {
    file: "10_混合形态自适应局部校正.svg",
    title: "混合形态自适应局部改正",
    group: "中间结果",
    note: "融合基础像素偏移和形态自适应分支，形成覆盖率更高的阶段性局部改正结果。",
  },
  {
    file: "11_混合像素偏移建筑高度图.svg",
    title: "混合分支建筑高度图",
    group: "阶段高度图",
    note: "展示混合分支对应的阶段性建筑高度分布，供后续严格重投影和联合筛选比较。",
  },
  {
    file: "12_纯影像特征局部配准.svg",
    title: "纯图像特征二维配准",
    group: "中间结果",
    note: "在二维局部窗口内自由搜索建筑特征，不使用高度方向约束，提供独立图像候选。",
  },
  {
    file: "13_纯影像特征建筑高度图.svg",
    title: "纯图像特征分支高度图",
    group: "阶段高度图",
    note: "把纯图像二维位移转换为阶段性高度结果，用于观察覆盖率和异常分布，不作为最终主图。",
  },
  {
    file: "14_纯影像特征配准审计.svg",
    title: "纯图像配准质量审计",
    group: "质量诊断",
    note: "检查边缘命中、连续率、位移幅度和置信度，为扩窗恢复与严格几何筛选提供依据。",
  },
  {
    file: "15_数量质量联合配准.svg",
    title: "数量-质量联合优化配准",
    group: "当前主结果",
    note: "联合严格距离-多普勒重投影、边界候选恢复、空间残差场和多分支筛选，是当前推荐配准结果。",
  },
  {
    file: "16_数量质量联合建筑高度图.svg",
    title: "最终建筑离地高度图",
    group: "当前主结果",
    note: "当前正式高度成果图。369栋建筑获得有效离地高度，灰色建筑保持无值，没有用Shapefile高度填充。",
  },
];

function run(text, options = {}) {
  return new TextRun({
    text,
    font: options.font || FONT,
    size: options.size || 21,
    color: BLACK,
    bold: Boolean(options.bold),
    italics: Boolean(options.italics),
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
    children: [run(text, options)],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    keepNext: true,
    children: [run(text, { bold: true, size: 32 })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    keepNext: true,
    children: [run(text, { bold: true, size: 26 })],
  });
}

function equation(formula, number) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 140, line: 340 },
    keepNext: true,
    children: [
      run(formula, { font: "Cambria Math", size: 21 }),
      run(`    (${number})`, { font: "Cambria Math", size: 20 }),
    ],
  });
}

function formula(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 100, after: 140 },
    keepNext: true,
    children: [run(text, { font: "Cambria Math", size: 21 })],
  });
}

function note(title, text) {
  return new Paragraph({
    spacing: { before: 110, after: 150, line: 330 },
    border: {
      top: { style: BorderStyle.SINGLE, size: 5, color: BLACK, space: 7 },
      bottom: { style: BorderStyle.SINGLE, size: 5, color: BLACK, space: 7 },
    },
    children: [
      run(`${title}：`, { bold: true }),
      run(text),
    ],
  });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function cell(text, width, options = {}) {
  const borders = options.header
    ? { top: thick, bottom: thin, left: none, right: none }
    : options.last
      ? { top: none, bottom: thick, left: none, right: none }
      : { top: none, bottom: none, left: none, right: none };
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    verticalAlign: VerticalAlign.CENTER,
    shading: { fill: "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 95, bottom: 95, left: 125, right: 125 },
    children: [
      new Paragraph({
        alignment: options.alignment || AlignmentType.LEFT,
        spacing: { after: 0, line: 280 },
        children: [run(String(text), { bold: Boolean(options.header), size: 19 })],
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
        tableHeader: true,
        children: headers.map((value, i) => cell(value, widths[i], { header: true })),
      }),
      ...rows.map(
        (row, rowIndex) =>
          new TableRow({
            children: row.map((value, i) =>
              cell(value, widths[i], { last: rowIndex === rows.length - 1 }),
            ),
          }),
      ),
    ],
  });
}

function pngSize(file) {
  const data = fs.readFileSync(file);
  if (data.toString("ascii", 1, 4) !== "PNG") {
    throw new Error(`Not a PNG: ${file}`);
  }
  return { width: data.readUInt32BE(16), height: data.readUInt32BE(20) };
}

function renderAssets() {
  fs.mkdirSync(ASSET_DIR, { recursive: true });
  for (const figure of FIGURES) {
    const source = path.join(PICALL, figure.file);
    const output = path.join(ASSET_DIR, figure.file.replace(/\.svg$/i, ".png"));
    if (!fs.existsSync(source)) {
      throw new Error(`Missing PICALL figure: ${source}`);
    }
    if (!fs.existsSync(output)) {
      throw new Error(
        `Missing rendered report asset: ${output}. Render the PICALL SVG files with CairoSVG first.`,
      );
    }
    figure.png = output;
  }
}

function imageBlock(file) {
  const size = pngSize(file);
  const maxWidth = 635;
  const maxHeight = 610;
  const scale = Math.min(maxWidth / size.width, maxHeight / size.height, 1);
  const width = Math.round(size.width * scale);
  const height = Math.round(size.height * scale);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 100 },
    children: [
      new ImageRun({
        type: "png",
        data: fs.readFileSync(file),
        transformation: { width, height },
        altText: {
          title: path.basename(file),
          description: "PICALL独立结果图",
          name: path.basename(file),
        },
      }),
    ],
  });
}

renderAssets();

const children = [];
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1050, after: 260 },
    children: [run("像素偏移建筑高度估计", { bold: true, size: 48 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 380 },
    border: {
      bottom: { style: BorderStyle.SINGLE, size: 6, color: BLACK, space: 10 },
    },
    children: [run("方法原理与PICALL结果报告", { bold: true, size: 36 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 750, after: 150 },
    children: [run("详细方法公式  |  15张SVG逐图独立展示", { bold: true, size: 26 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 150 },
    children: [run("最终有效高度369栋  |  当前主图：图14、图15", { size: 22 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 900 },
    children: [run("2026-07-27", { size: 20 })],
  }),
  pageBreak(),
);

children.push(
  h1("1 先说清楚：像素偏移法到底怎样估高"),
  para(
    "这套方法不是从SAR像素的亮度直接读取建筑高度，而是观察屋顶在SAR影像中的位置。SAR从斜上方照射，建筑越高，屋顶相对地面轮廓通常越向近距方向移动。程序先在影像中找到最像屋顶的位置，再不断改变候选屋顶高程并重新投影；哪个高度的投影轮廓最贴近影像结构，就采用哪个高度。",
  ),
  h2("1.1 用“透明建筑模型”理解"),
  para(
    "可以想象把一个透明的三维建筑模型盖在SAR图上。模型太矮时，投影屋顶落在影像结构的一侧；把模型抬高，屋顶投影沿高度敏感方向移动。不断上下调节模型，直到投影屋顶与真实影像中的屋顶边缘、连续亮线和内外强度变化最吻合。",
  ),
  para(
    "这里有两个问题必须分开：第一，影像中的屋顶究竟在哪里；第二，这个位置对应多高的屋顶。前者由图像增强和二维配准解决，后者由严格距离-多普勒重投影解决。二维配准结果只是高度搜索的线索，不是最终高度。",
  ),
  h2("1.2 一栋建筑经过哪些步骤"),
  table(
    ["阶段", "通俗解释", "得到的量"],
    [
      ["1 基准投影", "把地面屋顶和候选高度屋顶分别投到SAR图上", "单位高度对应的二维移动方向"],
      ["2 图像定位", "在三景影像中寻找最像屋顶的边缘和亮线", "二维偏移δrow、δcol"],
      ["3 偏移分解", "区分高度造成的移动和横向配准误差", "线性高度初值与横向残差"],
      ["4 严格搜索", "逐个改变屋顶高程并重新执行成像几何", "最优屋顶绝对高程"],
      ["5 质量控制", "检查形状、多景一致性、残差和边界解", "高度、置信等级或拒绝"],
    ],
    [1900, 5000, 2700],
  ),
  h2("1.3 输入与输出"),
  table(
    ["类别", "内容"],
    [
      ["输入", "三景共注册RSLC、建筑屋顶轮廓、轨道和成像参数"],
      ["直接观测", "投影屋顶与影像屋顶结构之间的二维像素偏移"],
      ["待求量", "屋顶绝对高程Z_roof"],
      ["最终输出", "建筑离地高度、置信等级、来源分支和残差"],
      ["当前覆盖", "1028栋中369栋有值，659栋保持空值"],
    ],
    [2300, 7300],
  ),
  h2("1.4 高度口径与使用边界"),
  para(
    "当前结果以4 m绝对高程作为统一地面基底，建筑离地高度等于最终屋顶绝对高程减4 m。Shapefile height字段用于建立初始投影和高度-像素位移关系，不用于填充最终缺失值；659栋无可靠解建筑继续保持空值。",
  ),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 110, after: 150 },
    children: [run("final_height_m = final_roof_elevation_m - 4.000 m", { font: "Cambria Math", size: 22 })],
  }),
  para(
    "当前高、中、低和补充级属于内部图像与几何质量等级。由于尚无覆盖全区的独立LiDAR、GNSS或实测建筑高度，本报告不把这些等级解释为外部真实精度，也不报告全区真实MAE或RMSE。",
  ),
  note(
    "一句话结论",
    "像素偏移法先用影像回答“屋顶在哪里”，再用严格成像几何回答“这个位置对应多高”；不能把像素总位移直接乘一个全区固定比例当成最终高度。",
  ),
);

children.push(
  pageBreak(),
  h1("2 方法概览：先在影像中找屋顶，再用几何求高度"),
  para(
    "这套方法不是直接从SAR亮度读取建筑高度。SAR影像中的一栋楼会因侧视成像发生位置偏移，而且屋顶边缘、墙面亮线和散斑会混在一起。因此，处理被拆成两个相互衔接的问题：先回答“建筑屋顶在SAR图像的哪一处”，再回答“这个位置对应多高的屋顶”。",
  ),
  h2("2.1 一条通俗的处理链"),
  table(
    ["步骤", "实际做法", "解决的问题"],
    [
      ["1 建立基准投影", "分别以0 m和Shapefile高度投影全部屋顶", "得到高度改变时屋顶在SAR中的移动方向"],
      ["2 增强SAR特征", "三景分别去斑、增强局部对比并提取多尺度边缘", "让屋顶边缘比散斑更容易识别"],
      ["3 自由二维搜索", "先2 pixel粗搜，再0.25 pixel细化", "找到最符合影像证据的屋顶平移位置"],
      ["4 三景融合", "每景独立评分，最终取中位分数", "降低单景偶然强散射影响"],
      ["5 几何分解", "把二维位移分成高度方向和垂直方向", "区分高度信号与横向配准误差"],
      ["6 严格高程搜索", "每个候选高程都重新执行距离-多普勒投影", "避免把线性像素偏移直接当最终高度"],
      ["7 多分支融合", "严格解优先，可靠历史混合解作为补充", "兼顾质量与覆盖数量"],
      ["8 质量控制", "拒绝边界解、多景不一致、残差过大和重复目标", "只保留可解释、可追溯的结果"],
    ],
    [1500, 4100, 4000],
  ),
  h2("2.2 为什么需要三景"),
  para(
    "三景RSLC已经共注册，因此同一个SAR像素位置可以直接比较。对每个候选屋顶位置，三景分别计算匹配分数，再取中位数。若只有一景亮、另两景不支持，该候选通常不会成为稳定峰值；若至少两景给出接近的位置，则结果更可信。",
  ),
  equation("S(fused, Δ) = median［S₁(Δ), S₂(Δ), S₃(Δ)］", 1),
  note(
    "直观理解",
    "可以把三景看成三位观察者。最终位置不听最极端的一位，而采用三者中间的意见；同时还要求至少一对观察者给出的最佳位置足够接近。",
  ),
);

children.push(
  pageBreak(),
  h1("3 严格投影与高度基准"),
  h2("3.1 为什么地图轮廓不能直接叠到SAR影像"),
  para(
    "SAR是侧视雷达。建筑越高，屋顶在距离向和方位向的投影位置越可能偏离地面轮廓。这个偏移不仅由高度决定，还与卫星位置、速度、斜距、多普勒频率和成像时刻有关。因此不能简单把地图轮廓平移到影像上，也不能假定所有建筑共享一个固定的“每米移动像素数”。",
  ),
  h2("3.2 建筑高度与像素偏移的近似关系"),
  para(
    "先用一个简化侧视模型理解高度为什么会形成像素偏移。若局部入射角θ从竖直方向量起，建筑高度为H，则屋顶相对底部在地距方向上的位移近似为H/tanθ。在原始斜距影像中，屋顶升高会使斜距缩短，变化量近似为-Hcosθ。",
  ),
  formula("Δx_ground ≈ H / tanθ"),
  formula("ΔR ≈ −Hcosθ，    Δcol ≈ ΔR / δR_pixel"),
  para(
    "Δx_ground是地距位移，ΔR是斜距变化，δR_pixel是斜距像元间隔。两种表达描述同一侧视效应，但坐标系和像元间隔不同，不能混着使用。",
  ),
  note(
    "示意计算",
    "假设θ=35°、建筑高30 m，则地距位移约为42.8 m；若地距像元为1.5 m，约移动28.6像素。若斜距像元为0.9 m，斜距变化约为-24.6 m，对应-27.3像素。该例只解释规律，不是当前单栋建筑的正式结果。",
  ),
  h2("3.3 为什么不能直接用“偏移像素×固定比例”"),
  para(
    "全局轨道误差、时间或数据解析误差、建筑轮廓平面误差和局部散射中心变化都会形成像素偏移，其中只有沿高度敏感方向的部分才能解释为高度。不同建筑的入射角、方向和位置也略有不同，所以固定比例只能用于粗略理解，不能作为正式反演。",
  ),
  h2("3.4 距离-多普勒投影"),
  para(
    "对建筑屋顶顶点X，需要寻找成像时刻t和斜距R，使它既满足到卫星位置S(t)的距离条件，也满足卫星速度V(t)对应的多普勒条件。求解后，t和R被换算为SAR方位行和距离列。",
  ),
  equation("距离条件：‖X − S(t)‖₂ − R = 0", 2),
  equation("多普勒条件：[X − S(t)]·V(t) − C_D(R, f_D, λ) = 0", 3),
  para(
    "式中f_D为多普勒频率，λ为雷达波长，C_D表示由成像参数确定的多普勒项。当前流程只投影建筑顶面，不进行墙面拉伸。全局雷达坐标改正为方位行+34 pixel、距离列-1 pixel；局部配准只处理逐建筑剩余误差。",
  ),
  h2("3.5 两次严格投影得到逐建筑“像素/米”向量"),
  para(
    "同一栋建筑至少计算两个严格投影：0 m绝对高程投影，以及Shapefile height对应的绝对高程投影。两者质心之差除以高程差，得到该建筑局部的每米像素位移向量d。",
  ),
  equation("d = (d_col, d_row) = [c(H₂) − c(H₁)] / (H₂ − H₁)", 4),
  para(
    "向量d=[d_col,d_row]的单位是像素/米，给出“这栋建筑的屋顶升高1 m时，在当前场景中大致往哪里移动”。它是逐建筑、逐场景计算的，不是全区常数。",
  ),
  para(
    "若图像匹配得到二维偏移Δp=[Δcol,Δrow]，把它投影到d方向可得到线性屋顶高程初值；剩余的垂直分量主要用于描述局部配准残差。",
  ),
  formula("Z_linear = (d · Δp) / (d · d)"),
  formula("Δp_perp = Δp − d·Z_linear"),
  para(
    "例如d=[-0.80,0.05]像素/米、匹配偏移Δp=[-16,1]像素时，线性高度初值约为20 m。与高度方向不一致的那部分偏移会单独保留，不会全部塞进建筑高度。",
  ),
  h2("3.6 高度基准"),
  equation("建筑离地高度 H = 屋顶绝对高程 Z_roof − 4.000 m", 5),
  note(
    "必须区分",
    "严格投影搜索的是屋顶绝对高程，地图和最终表中的final_height_m是离地高度。旧流程把二者混用会整体偏高约4 m。Shapefile高度只参与初始化，不用于填补无解建筑。",
  ),
);

children.push(
  pageBreak(),
  h1("4 SAR特征增强：让屋顶边缘从散斑中显出来"),
  h2("4.1 对数压缩与局部保边去斑"),
  para(
    "原始SAR幅度动态范围很大，少数强亮点可能淹没普通屋顶边缘。流程先做对数压缩，再在5×5邻域估计局部均值和方差。平坦区域更接近局部均值，边缘和真实强散射处保留更多原值，这是一种类似Lee滤波的保边去斑。",
  ),
  equation("A_log = log[1 + 20·max(A,0)] / log(21)", 6),
  equation("A_lee = μ₅ + w_lee(A_log − μ₅)", 7),
  equation("w_lee = clip[(σ₅² − σ_n²) / max(σ₅²,10⁻⁶), 0, 1]", 8),
  para(
    "σ_n²取局部方差的第35百分位，代表常见散斑水平。若某处方差接近噪声，w_lee较小，输出更平滑；若某处是明显边缘，w_lee较大，结构被保留。",
  ),
  h2("4.2 局部对比与多尺度边缘"),
  para(
    "仅去斑仍不足以突出建筑。流程再用σ=4 pixel的宽尺度背景计算局部标准化对比，并把它与保边幅度按0.32和0.68融合。随后分别在0.65和1.45 pixel尺度计算Sobel边缘，小尺度保留细节，大尺度保持连续结构。",
  ),
  equation("A_enh = N［0.68·N(A_lee) + 0.32·N(Z_local)］", 9),
  equation("E = N［0.65·N(E_fine) + 0.35·N(E_coarse)］", 10),
  para(
    "N表示分位数稳健归一化。增强数组只参与候选评分，不写回原始RSLC，也不改变SAR几何和最终统计。三景分别增强后，再对增强幅度和边缘响应取逐像素中位数用于展示。",
  ),
  note(
    "通俗理解",
    "先压低“特别亮但不一定是屋顶”的少数像素，再保留成片、连续并与建筑轮廓方向一致的边缘。目标不是把图变得好看，而是让匹配分数更可靠。",
  ),
);

children.push(
  pageBreak(),
  h1("5 形态自适应二维配准"),
  h2("5.1 为什么允许先自由二维搜索"),
  para(
    "真实局部误差不一定完全沿高度方向。轨道、建筑轮廓、全局改正和局部散射中心都可能带来横向偏差。因此，先让建筑屋顶在二维窗口内自由搜索，不使用高度先验惩罚；找到影像支持的位置后，再把位移交给严格几何解释。",
  ),
  h2("5.2 四类影像证据"),
  table(
    ["证据", "计算方式", "直观含义"],
    [
      ["方向一致边缘 E_o", "边缘强度乘以梯度与轮廓法向一致度", "边缘不但要强，方向还应像屋顶边界"],
      ["边缘连续率 C", "轮廓采样点中E≥0.38的加权比例", "避免只靠一个亮点命中"],
      ["内外对比 K", "屋顶内部均值减去外部背景均值", "屋顶区域应与周围不同"],
      ["内部亮散射 B", "屋顶内部增强幅度第82百分位", "允许屋顶存在稳定强散射中心"],
    ],
    [2200, 4000, 3400],
  ),
  equation("S_s(Δ) = w_e·Z(E_o) + w_c·Z(C) + w_k·Z(K) + w_b·Z(B)", 11),
  para(
    "Z表示稳健标准化，s表示场景，Δ=(Δcol,Δrow)表示候选平移。不同建筑使用不同权重：长条建筑更重视边缘方向和连续性，小建筑适当提高亮散射权重，大建筑要求更多有效像素。",
  ),
  h2("5.3 粗搜、细化和峰值可信度"),
  para(
    "搜索先以2 pixel步长扫描整个局部窗口，再以粗搜最佳点为中心，在±2 pixel范围用0.25 pixel步长细化。最佳峰值必须明显优于距离它至少1.5 pixel的次优峰。",
  ),
  equation("Margin = S(Δ_best) − max［S(Δ), ‖Δ−Δ_best‖≥1.5］", 12),
  para(
    "同时要求：有效采样比例不少于90%，边缘强度和连续率中位数均不少于0.10；最接近的两景最佳位移差不超过4 pixel，融合位置到该景对中心不超过3 pixel；最佳点不能贴住搜索边界。命中边界的建筑按形态扩窗到16-28 pixel后再尝试恢复。",
  ),
);

children.push(
  pageBreak(),
  h1("6 从二维像素偏移到严格建筑高度"),
  h2("6.1 先分解高度方向与横向残差"),
  para(
    "设零高程屋顶投影质心为c₀，图像匹配目标质心为c_t，局部每米位移向量为d。把二维偏移Δc=c_t−c₀投影到d方向，可以得到屋顶绝对高程的初始估计；与d垂直的分量更可能是局部配准残差。",
  ),
  equation("Z₀ = (Δcol·d_col + Δrow·d_row) / (d_col² + d_row²)", 13),
  equation("p = (−d_row, d_col) / ‖d‖₂", 14),
  para(
    "流程利用高质量控制点估计局部空间横向残差场，并把每栋建筑的垂直修正限制在±3 pixel。这样，沿高度方向的位移主要解释为高程，横向系统误差则先被校正。",
  ),
  h2("6.2 每个候选高程都重新严格投影"),
  para(
    "初始高度Z₀只是搜索中心。程序在Z₀附近±15 m范围内，先以1 m步长生成候选屋顶绝对高程；每个候选都重新求解距离-多普勒方程并生成屋顶多边形P(Z)。粗搜最佳值附近再以0.1 m步长细化。",
  ),
  equation("J(Z) = d_centroid［P(Z),T］ + 0.15·d_Hausdorff［P(Z),T］", 15),
  equation("Z_roof = arg min_Z J(Z)", 16),
  para(
    "质心距离保证整体位置接近，Hausdorff距离约束最不匹配的边界，系数0.15避免少量轮廓异常完全主导结果。最终离地高度再由式(5)减去4 m基底。",
  ),
  note(
    "为什么比线性换算可靠",
    "线性换算假定屋顶随高度沿一条固定直线移动；严格重投影会重新考虑卫星轨道、斜距、多普勒和建筑各顶点，因此最终高度与SAR成像几何保持一致。",
  ),
);

children.push(
  pageBreak(),
  h1("7 联合筛选、置信度与最终结果"),
  h2("7.1 严格分支的置信度"),
  table(
    ["等级", "严格几何条件", "解释"],
    [
      ["高置信", "J≤2 pixel，且最接近两景差≤2.5 pixel", "几何残差小，多景位置稳定"],
      ["中置信", "J≤4 pixel", "总体匹配可靠，但精细边界存在一定偏差"],
      ["低置信", "J≤7 pixel", "仍有几何支持，使用时需同时查看诊断"],
      ["拒绝", "J>7 pixel或高程搜索命中边界", "不输出正式高度"],
    ],
    [1700, 4300, 3600],
  ),
  h2("7.2 多分支融合"),
  para(
    "严格重投影分支优先采用。若严格分支与已有混合分支的屋顶绝对高程相差不超过3 m，中置信可升级为高置信、低置信可升级为中置信；若低置信分支差异大于8 m，则降为补充级。严格分支无解时，只允许此前已经独立通过质量控制的混合结果以补充级进入最终表。",
  ),
  equation("Agreement = |Z_strict − Z_hybrid|", 17),
  para(
    "最终还会检查两个建筑是否占用了几乎相同的SAR目标。发生重复时，优先保留置信等级更高、峰值间隔更大的建筑，较弱结果被拒绝。高度小于3 m、搜索边界解、无有限值或使用先验填充的结果均不能通过。",
  ),
  h2("7.3 当前数量与解释边界"),
  table(
    ["阶段", "数量"],
    [
      ["建筑总数", "1028"],
      ["原纯图像二维配准通过", "322"],
      ["扩窗恢复候选", "100"],
      ["联合严格候选", "422"],
      ["严格几何可用候选", "187"],
      ["最终严格分支采用", "185"],
      ["历史混合分支补充", "184"],
      ["最终有值建筑", "369"],
      ["最终无值建筑", "659"],
    ],
    [6100, 3500],
  ),
  note(
    "精度边界",
    "当前高、中、低和补充级是内部图像与几何质量，不是外部真实高度精度。没有覆盖全区的独立LiDAR、GNSS或实测控制，因此不能把置信度直接写成真实MAE或RMSE。",
  ),
);

FIGURES.forEach((figure, index) => {
  const figureNumber = String(index + 1).padStart(2, "0");
  const displayFile = `${figureNumber}_${figure.file.slice(3)}`;
  children.push(
    pageBreak(),
    h1(`图${figureNumber}  ${figure.title}`),
    new Paragraph({
      spacing: { after: 90 },
      children: [
        run(`类别：${figure.group}`, { bold: true }),
        run(`    源文件：${displayFile}`, { size: 18 }),
      ],
    }),
    para(figure.note, { indent: false }),
    imageBlock(figure.png),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 40, after: 0 },
      children: [run(`图${Number(figureNumber)}  ${figure.title}`, { size: 18 })],
    }),
  );
});

const doc = new Document({
  creator: "Codex",
  title: "像素偏移建筑高度估计方法原理与PICALL结果报告",
  description: "详细方法、公式、质量控制及15张PICALL SVG逐图独立展示",
  styles: {
    default: {
      document: { run: { font: FONT, size: 21, color: BLACK } },
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: BLACK },
        paragraph: { spacing: { before: 220, after: 180 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: BLACK },
        paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 1 },
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
              children: [run("像素偏移建筑高度估计方法原理与PICALL结果报告", { size: 16 })],
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
                run("第 ", { size: 16 }),
                new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: BLACK }),
                run(" 页", { size: 16 }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUTPUT, buffer);
  process.stdout.write(`${OUTPUT}\n`);
});
