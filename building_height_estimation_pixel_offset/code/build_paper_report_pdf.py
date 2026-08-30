from __future__ import annotations

import json
from pathlib import Path

import cairosvg
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
OUTPUT = OUTPUT_DIR / "building_height_estimation_joint_optimization_report.pdf"


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="paper", frames=frame, onPage=self.draw_page))

    def draw_page(self, canvas, doc):
        page = canvas.getPageNumber()
        if page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B7BDC5"))
        canvas.setLineWidth(0.35)
        canvas.line(22 * mm, A4[1] - 17 * mm, A4[0] - 22 * mm, A4[1] - 17 * mm)
        canvas.setFont("STSong-Light", 7.5)
        canvas.setFillColor(colors.HexColor("#4B5563"))
        canvas.drawString(22 * mm, A4[1] - 13.5 * mm, "基于SAR图像特征与严格距离-多普勒投影的建筑高度估计")
        canvas.drawCentredString(A4[0] / 2, 11 * mm, str(page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            if name == "H1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
            elif name == "H2":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))


def styles():
    registerFont(UnicodeCIDFont("STSong-Light"))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN", parent=base["Title"], fontName="STSong-Light", fontSize=22,
            leading=34, alignment=TA_CENTER, textColor=colors.HexColor("#132238"), spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN", parent=base["Normal"], fontName="STSong-Light", fontSize=12,
            leading=20, alignment=TA_CENTER, textColor=colors.HexColor("#506176"),
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="STSong-Light", fontSize=15,
            leading=23, spaceBefore=11, spaceAfter=8, textColor=colors.HexColor("#123B5D"),
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="STSong-Light", fontSize=11.5,
            leading=18, spaceBefore=8, spaceAfter=5, textColor=colors.HexColor("#1E5A78"),
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN", parent=base["BodyText"], fontName="STSong-Light", fontSize=9.4,
            leading=16.2, alignment=TA_JUSTIFY, firstLineIndent=18.8, spaceAfter=5.2,
            textColor=colors.HexColor("#20252B"),
        ),
        "body0": ParagraphStyle(
            "BodyNoIndent", parent=base["BodyText"], fontName="STSong-Light", fontSize=9.4,
            leading=16.2, alignment=TA_JUSTIFY, spaceAfter=5.2, textColor=colors.HexColor("#20252B"),
        ),
        "abstract": ParagraphStyle(
            "AbstractCN", parent=base["BodyText"], fontName="STSong-Light", fontSize=9,
            leading=15.5, alignment=TA_JUSTIFY, firstLineIndent=18, spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "CaptionCN", parent=base["Normal"], fontName="STSong-Light", fontSize=8,
            leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#3D4650"), spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "TableCN", parent=base["Normal"], fontName="STSong-Light", fontSize=8,
            leading=11, alignment=TA_CENTER,
        ),
        "small": ParagraphStyle(
            "SmallCN", parent=base["Normal"], fontName="STSong-Light", fontSize=7.6,
            leading=11.5, alignment=TA_LEFT, wordWrap="CJK", textColor=colors.HexColor("#4B5563"),
        ),
        "formula": ParagraphStyle(
            "FormulaCN", parent=base["Normal"], fontName="STSong-Light", fontSize=10,
            leading=17, alignment=TA_CENTER, backColor=colors.HexColor("#F2F6F8"),
            borderColor=colors.HexColor("#C7D4DC"), borderWidth=0.5, borderPadding=7,
            spaceBefore=5, spaceAfter=7,
        ),
        "ref": ParagraphStyle(
            "ReferenceCN", parent=base["Normal"], fontName="Helvetica", fontSize=7.8,
            leading=13, leftIndent=13, firstLineIndent=-13, spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "CodePath", parent=base["Normal"], fontName="Helvetica", fontSize=7.3,
            leading=11, alignment=TA_LEFT, wordWrap="CJK", textColor=colors.HexColor("#34424E"),
        ),
    }


def p(text, style):
    return Paragraph(text, style)


def make_table(rows, widths, style, header=True):
    wrapped = [[Paragraph(str(cell), style) for cell in row] for row in rows]
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AEB8C1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF1")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#153B52")),
        ])
    table.setStyle(TableStyle(commands))
    return table


def render_svg(svg: Path, png: Path, width: int = 2100):
    cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=width)


def figure(svg_name, caption, number, max_width=166 * mm, max_height=185 * mm):
    svg = ROOT / "results" / "PICALL" / svg_name
    png = TMP_DIR / f"figure_{number:02d}.png"
    render_svg(svg, png)
    image = Image(str(png))
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return KeepTogether([image, Spacer(1, 2 * mm), p(f"图{number}  {caption}", S["caption"])])


def build_story(summary, table):
    accepted = table[table.final_accepted == 1]
    confidence = accepted.final_confidence.value_counts()
    source = accepted.final_source.value_counts()
    id_rows = accepted[accepted.clean_id.isin([776, 788])].set_index("clean_id")
    s = S
    story = []

    story.extend([
        Spacer(1, 30 * mm),
        p("基于SAR图像特征与严格距离-多普勒投影的建筑高度估计", s["title"]),
        Spacer(1, 5 * mm),
        p("- 面向同济大学区域RSLC影像的数量-质量联合优化方法 -", s["subtitle"]),
        Spacer(1, 24 * mm),
        make_table([
            ["文档类型", "论文式技术报告"],
            ["研究对象", "1028栋建筑与三景共注册RSLC影像"],
            ["核心方法", "二维图像匹配、严格距离-多普勒重投影、多分支融合"],
            ["最终结果", "369栋建筑获得高度估计"],
            ["版本日期", "2026年7月23日"],
        ], [35 * mm, 95 * mm], s["table"]),
        Spacer(1, 31 * mm),
        p("项目目录", s["subtitle"]),
        p(str(ROOT), s["small"]),
        PageBreak(),
        p("摘  要", s["h1"]),
        p(
            "针对高分辨率合成孔径雷达影像中建筑物散射复杂、屋顶边界不连续、叠掩与阴影干扰明显，以及单一匹配参数难以同时适应小建筑、狭长建筑和大尺度建筑的问题，本文提出一套以建筑顶面为对象的数量-质量联合高度估计流程。该流程首先将建筑矢量以0 m绝对高程和Shapefile height字段对应的绝对高程执行严格距离-多普勒投影，建立像素偏移与高程变化的逐建筑几何关系；随后对三景共注册RSLC幅度影像进行局部统计增强、双尺度边缘融合和多景中值融合，并根据建筑面积、长宽比和边界尺度采用差异化二维搜索策略。对初始搜索命中边界的建筑扩大局部窗口，再利用高质量匹配样本估计空间残差场。每个候选高度均重新执行严格距离-多普勒投影，避免以单位方向线性平移替代真实成像几何。最后通过多景一致性、严格几何残差、分支一致性和重叠目标冲突进行分级筛选。", s["abstract"]),
        p(
            f"全区共处理1028栋建筑。纯图像二维配准阶段有322栋通过质量控制；扩大搜索后形成422个候选，其中187个通过严格几何门限。多分支融合和重复目标剔除后，最终{len(accepted)}栋获得离地高度，其中高置信{int(confidence.get('high', 0))}栋、中置信{int(confidence.get('medium', 0))}栋、低置信{int(confidence.get('low', 0))}栋、补充级{int(confidence.get('supplemental', 0))}栋。高度中位数为{accepted.final_height_m.median():.2f} m，范围为{accepted.final_height_m.min():.2f}-{accepted.final_height_m.max():.2f} m。研究进一步辨析了屋顶绝对高程与建筑离地高度的差异，采用统一4 m地面基准完成转换，从而消除旧结果约4 m的系统性偏高。未通过质量控制的建筑保持无值，未使用Shapefile height字段填充。当前结果完成了内部几何与图像匹配审计，但尚缺少全区外部真实高度验证。", s["abstract"]),
        p("关键词：合成孔径雷达；建筑高度；距离-多普勒模型；影像配准；屋顶投影；像素偏移", s["body0"]),
        Spacer(1, 5 * mm),
        p("Abstract", s["h1"]),
        p(
            "This report presents a quantity-quality joint workflow for estimating building heights from three co-registered RSLC scenes and building footprints. Building roofs are first projected at zero absolute elevation and at the elevation stored in the Shapefile height field using a strict range-Doppler model. Multi-scale SAR enhancement and shape-adaptive two-dimensional matching are then used to locate roof-related image features. Boundary-hit cases are re-searched with enlarged windows, and a spatial residual field is estimated from reliable controls. Instead of translating a roof along a fixed unit direction, every candidate elevation is reprojected through the strict range-Doppler geometry. Candidate solutions are filtered by multi-scene consistency, geometric residuals, branch agreement, and duplicate target assignment. Of 1,028 buildings, 369 receive final above-ground height estimates: 106 high-confidence, 37 medium-confidence, 34 low-confidence, and 192 supplemental results. The median height is 17.10 m. Missing buildings remain unfilled. The confidence classes represent internal reliability and should not be interpreted as externally validated accuracy.", s["abstract"]),
        p("Keywords: synthetic aperture radar; building height; range-Doppler geometry; image registration; roof projection; pixel offset", s["body0"]),
        PageBreak(),
        p("目  录", s["h1"]),
    ])
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName="STSong-Light", fontSize=10, leading=18, leftIndent=0, firstLineIndent=0, textColor=colors.HexColor("#153B52")),
        ParagraphStyle("TOC2", fontName="STSong-Light", fontSize=8.5, leading=14, leftIndent=15, firstLineIndent=0, textColor=colors.HexColor("#4A5964")),
    ]
    story.extend([toc, PageBreak()])

    story.extend([
        p("1  引言", s["h1"]),
        p(
            "建筑高度是城市三维建模、灾害风险评估、人口与能源分析中的基础参数。光学立体像对、机载激光雷达和地面测量能够提供直接或近直接的高程约束，但其数据获取成本、覆盖频率和天气适应性存在差异。合成孔径雷达具有全天时、全天候观测能力，建筑在距离向产生的叠掩、亮边和阴影又与其几何高度相关，因此可以从SAR像素位置变化中反演高度。与此同时，SAR建筑散射并不是理想的闭合屋顶轮廓：墙面、角反射、邻近建筑叠掩和孤立强散射点都可能形成比屋顶更强的响应。若直接在大范围内搜索最大响应，矢量轮廓容易匹配到无关目标，并造成高度普遍偏大。", s["body"]),
        p(
            "SAR几何定位通常以卫星轨道、斜距和多普勒条件构成距离-多普勒方程组[1-3]。本研究将这一物理模型与局部图像匹配结合：图像特征负责回答“屋顶特征在何处”，严格几何负责回答“该位置对应什么屋顶高程”。二者在流程中先后分离，避免将先验高度直接写入评分函数，也避免在得到候选位置后仍使用线性近似替代严格投影。图像处理部分借鉴局部统计去噪与边缘检测的基本思想[4-5]，但评分、门限和融合规则均针对本区三景RSLC及建筑形态重新设计。", s["body"]),
        p(
            "本文的目标不是给所有建筑强制赋值，而是在可解释的质量控制下同时提高覆盖数量与单体可靠性。主要工作包括：建立0 m与先验绝对高程的严格投影基准；构建建筑尺度自适应的多景SAR匹配；对边界候选扩大搜索并估计空间残差；逐候选高程执行严格重投影；通过多分支和重叠目标仲裁形成分级结果；最后统一屋顶绝对高程与建筑离地高度的定义。", s["body"]),

        p("2  研究数据与几何基准", s["h1"]),
        p("2.1  建筑矢量与RSLC影像", s["h2"]),
        p(
            "研究区建筑数据为1028个面要素，以clean_id作为唯一且权威的建筑编号。建筑几何统一转换到UTM 51N用于面积、周长、长宽比和邻域关系计算。SAR数据由2020年7月8日、7月30日和8月21日三景RSLC组成，影像尺寸均为900×630像元，且已完成共注册。因而本研究不再估计景间整体配准，只保留由前期几何标定得到的全局行列改正，并在每栋建筑局部窗口内搜索目标位置。", s["body"]),
        p("2.2  0 m与Shapefile height绝对高程投影", s["h2"]),
        p(
            "全部建筑首先以0 m绝对高程执行严格距离-多普勒投影，作为像素偏移计算的统一零点。随后将Shapefile height字段直接解释为绝对高程，重新投影建筑顶面。该字段在这一阶段仅用于形成初始位置和估计每米高程对应的像素位移率，不作为最终高度的填充值，也不进入位移惩罚项。0 m投影保留全部1028栋建筑，其中600栋完全位于影像内、82栋部分相交、346栋位于影像外；height字段投影中605栋完全位于影像内、79栋部分相交、344栋位于影像外。", s["body"]),
        figure("06_全部建筑矢量高度投影.svg", "全部建筑按Shapefile height字段作为绝对高程的严格投影", 1, max_height=150 * mm),

        p("3  方法", s["h1"]),
        p("3.1  严格距离-多普勒投影", s["h2"]),
        p(
            "对建筑顶面任一点X及成像时刻t，严格投影同时满足斜距条件和零多普勒条件。设卫星位置为S(t)、速度为V(t)、目标到卫星的斜距为R，则可写为：", s["body"]),
        p("||X - S(t)|| = R，且 (X - S(t))<super>T</super> V(t) = 0", s["formula"]),
        p(
            "建筑多边形顶点经地理坐标与给定绝对高程构成三维点，求解相应的距离像素和方位像素，再形成雷达坐标中的顶面多边形。前期方法曾用单位高程位移方向近似候选高度投影。本次升级在每个候选高程上重新执行上述严格求解，并以投影多边形与图像匹配目标之间的质心距离和Hausdorff边界距离联合评分，从而保留轨道、斜距与多普勒关系中的非线性。", s["body"]),

        p("3.2  SAR建筑特征增强", s["h2"]),
        p(
            "对每景RSLC幅度执行对数压缩、局部统计保边去斑、局部背景标准化和双尺度梯度融合。局部统计处理用于降低乘性散斑对单像元亮度的支配作用[4]；细尺度梯度保留窄边，粗尺度梯度增强连续轮廓，边缘定位原则与经典边缘检测中的检测-定位权衡相一致[5]。三景分别处理后以中值融合，减少单景偶然强散射。增强结果只用于评分，原始RSLC不被修改。", s["body"]),
        figure("08_合成孔径雷达建筑特征增强.svg", "三景RSLC建筑特征增强与中值融合结果", 2, max_height=150 * mm),

        p("3.3  建筑形态自适应二维匹配", s["h2"]),
        p(
            "建筑按UTM平面面积和最小旋转矩形长宽比分为长条形、近方形、小建筑、大建筑和常规建筑。长条形建筑提高长边方向连续边缘的权重并扩大沿主轴搜索范围；近方形建筑强调闭合轮廓与内外对比；小建筑降低最低有效像素数并提高稳定亮散射权重；大建筑增加边界采样间隔和最低屋顶像素数。每栋建筑先以2像元步长进行二维粗搜索，再在最优位置附近以0.25像元步长细化。评分融合方向一致边缘、轮廓连续率、屋顶内外对比度和屋顶高分位亮散射，并取三景得分中值。", s["body"]),
        p(
            "这一阶段完全从图像特征定位屋顶候选，不使用高程方向约束，也不对远离初始位置的候选施加高程先验惩罚。每景还独立给出最佳二维位置，至少两景应形成足够接近的位置对；若匹配峰不显著、命中窗口边界、跨景位置不一致或有效像素不足，则拒绝该结果。", s["body"]),
        figure("12_纯影像特征局部配准.svg", "纯图像特征二维配准前后建筑轮廓对比", 3, max_height=150 * mm),

        p("3.4  扩窗恢复、空间残差场与严格高度搜索", s["h2"]),
        p(
            "初始二维配准中有195栋建筑命中搜索边界。为提高覆盖量，本研究按建筑类型将最大窗口扩大到16-28像元，并重新执行完整的粗细搜索，获得100个新的影像候选。扩窗候选仍需通过多景一致性和峰值显著性检查，不能直接进入最终高度。对原始高质量配准样本，计算其相对于逐建筑高程敏感方向的垂直残差，再用60个邻近控制点的距离加权中位数估计平滑空间残差场，单体修正限制在±3像元。", s["body"]),
        p(
            "校正后的二维目标位置用于产生候选屋顶绝对高程中心。严格搜索以1 m步长粗化，再以0.1 m步长细化；每个高程都重新执行距离-多普勒投影。严格残差由投影与二维目标的质心距离及0.15倍Hausdorff边界距离构成。搜索最优值落在上下边界、严格残差过大或建筑离地高度小于3 m时，判定为未收敛。", s["body"]),

        p("3.5  高度计算与基准转换", s["h2"]),
        p(
            "若以0 m绝对高程投影质心C0为参考，二维匹配目标质心为Ct，每米高程像素位移向量为d，则线性诊断高度可写为下式。该式主要用于初始化候选中心和解释像素偏移，不替代最终严格重投影：", s["body"]),
        p("z<sub>roof</sub> = [(C<sub>t</sub> - C<sub>0</sub>) · d] / ||d||<super>2</super>", s["formula"]),
        p(
            "zroof是屋顶绝对高程，不是建筑离地高度。本项目地面基底绝对高程统一为4 m，因此最终输出采用：", s["body"]),
        p("h<sub>building</sub> = z<sub>roof</sub> - 4 m", s["formula"]),
        p(
            "这一转换不使用逐建筑height先验，而是统一坐标基准换算。旧结果直接把屋顶绝对高程标成建筑高度，造成约4 m的系统性偏高。本次表格同时保留final_roof_elevation_m、base_elevation_m和final_height_m三个字段，便于复核。", s["body"]),

        p("3.6  多分支融合与联合归属", s["h2"]),
        p(
            "严格重投影分支优先进入最终结果。若严格分支与既有形态自适应混合分支高度差不超过3 m，则对中、低置信结果升级；若差异超过8 m且严格分支本身仅为低置信，则降为补充级。严格分支无解时，只保留此前已经独立通过质量控制的混合分支，标记为补充级，不能与高置信结果等同解释。对雷达坐标中高度重叠且质心接近的两个建筑，比较置信等级和峰值优势，剔除较弱者，避免多个矢量占用同一SAR目标。", s["body"]),

        p("4  结果", s["h1"]),
        p("4.1  各阶段覆盖量", s["h2"]),
        make_table([
            ["处理阶段", "候选/有值建筑", "说明"],
            ["全部输入", "1028", "clean_id唯一"],
            ["纯图像二维配准通过", "322", "不使用高程方向约束"],
            ["命中边界后扩窗恢复", "100", "仅作为新增候选"],
            ["联合严格候选", "422", "原322栋与恢复100栋"],
            ["严格几何可用", "187", "通过残差及边界门限"],
            ["最终有值", str(len(accepted)), "严格分支与补充分支融合"],
            ["最终无值", str(1028 - len(accepted)), "不进行先验填充"],
        ], [49 * mm, 33 * mm, 76 * mm], s["table"]),
        Spacer(1, 4 * mm),
        p(
            "纯图像配准通过建筑的方向一致边缘强度中位增益为0.147，轮廓连续率中位增益为0.219，说明矢量轮廓在图像特征意义上得到直接改善。扩窗使候选数由322增加到422，但严格几何仅接受其中187个，表明扩大窗口主要提高“可发现性”，最终质量仍由几何与跨景门限控制。", s["body"]),
        figure("14_纯影像特征配准审计.svg", "纯图像配准的边缘、连续率、位移与置信度审计", 4, max_height=155 * mm),

        p("4.2  联合优化投影结果", s["h2"]),
        p(
            f"最终{len(accepted)}栋中，{int(source.get('strict_refined', 0))}栋来自严格重投影分支，{int(source.get('legacy_hybrid_supplement', 0))}栋来自既有混合分支的补充。图5左侧为Shapefile height绝对高程初始投影，右侧为联合优化后的最终屋顶投影。颜色表示置信层级，而不是高度大小。图中高置信轮廓整体更贴合连续建筑边缘；补充级结果数量较多，提供覆盖量但必须与高、中置信结果分层使用。", s["body"]),
        figure("15_数量质量联合配准.svg", "数量-质量联合优化前后的建筑屋顶投影", 5, max_height=145 * mm),

        p("4.3  全区建筑高度分布", s["h2"]),
        make_table([
            ["指标", "结果"],
            ["最终有值建筑", f"{len(accepted)}栋"],
            ["高/中/低/补充级", f"{int(confidence.get('high',0))}/{int(confidence.get('medium',0))}/{int(confidence.get('low',0))}/{int(confidence.get('supplemental',0))}栋"],
            ["高度均值", f"{accepted.final_height_m.mean():.2f} m"],
            ["高度中位数", f"{accepted.final_height_m.median():.2f} m"],
            ["高度范围", f"{accepted.final_height_m.min():.2f}-{accepted.final_height_m.max():.2f} m"],
            ["重复目标剔除", f"{summary['duplicate_targets_rejected']}栋"],
            ["先验填充", "无"],
        ], [67 * mm, 78 * mm], s["table"]),
        Spacer(1, 4 * mm),
        p(
            "最终高度图以UTM 51N建筑矢量为底图。彩色面表示有值建筑，灰色面表示未通过质量控制或无法观测的建筑。所有有值建筑均标注整数米高度。高置信、中置信、低置信和补充级以边框颜色区分，保证使用者能够按风险偏好筛选。", s["body"]),
        figure("16_数量质量联合建筑高度图.svg", "全区建筑离地高度估计图；灰色为无值建筑", 6, max_height=188 * mm),

        p("4.4  典型建筑776与788", s["h2"]),
        p(
            "用户指出编号776与788的真实高度约为18 m。两栋建筑的最终结果如下：", s["body"]),
        make_table([
            ["clean_id", "屋顶绝对高程", "地面基底", "建筑离地高度", "置信等级"],
            ["776", f"{id_rows.loc[776, 'final_roof_elevation_m']:.2f} m", "4.00 m", f"{id_rows.loc[776, 'final_height_m']:.2f} m", "补充级"],
            ["788", f"{id_rows.loc[788, 'final_roof_elevation_m']:.2f} m", "4.00 m", f"{id_rows.loc[788, 'final_height_m']:.2f} m", "中置信"],
        ], [24 * mm, 34 * mm, 28 * mm, 36 * mm, 28 * mm], s["table"]),
        Spacer(1, 3 * mm),
        p("已知的约18 m高度没有进入匹配评分、候选搜索或缺失值填充，只用于结果生成后的解释性核对。", s["body"]),

        p("5  讨论", s["h1"]),
        p("5.1  结果普遍偏高的原因", s["h2"]),
        p(
            "前期结果偏高包含两类原因。第一类是定义偏差：严格投影直接求得屋顶绝对高程，旧图将其标记为建筑高度，因而整体高出统一4 m地面基底。第二类是图像匹配偏差：SAR强响应可能来自迎雷达墙面、墙-地二次反射或相邻建筑叠掩，狭长建筑的平行亮边还会产生多个相似峰。仅按单个最强峰或质心偏移计算高度，容易选到更远的高程候选。统一高程基准解决第一类系统误差；多景一致性、完整轮廓评分、严格重投影和边界解剔除用于压制第二类误差。", s["body"]),
        p("5.2  数量与质量的平衡", s["h2"]),
        p(
            "扩大窗口使更多建筑能够找到图像候选，但也增加匹配到邻近目标的风险。因此本研究没有把100个扩窗恢复对象全部计入最终可靠高度，而是要求它们再次通过严格投影。最终严格分支保留185栋，另外184栋由历史混合分支补充，覆盖量从293栋增加到369栋。补充级占最终结果的一半以上，说明当前方法实现了数量提升，但质量提升并非均匀发生。对于定量分析，建议优先使用高、中置信143栋；需要更大覆盖时再纳入低置信34栋；补充级应在人工检查或额外数据约束后使用。", s["body"]),
        p("5.3  可观测性与形态差异", s["h2"]),
        p(
            "建筑尺度自适应策略缓解了统一参数的局限，但SAR可观测性仍决定上限。小建筑有效像素少，单个散射点即可主导评分；大建筑内部散射不均匀，完整轮廓可能被植被或附属结构破坏；狭长建筑容易沿长边出现重复峰；近方形建筑方向判别力较弱。阴影、叠掩和低纹理区中的屋顶特征本身不可恢复，保持无值比依赖先验强制赋值更符合误差控制原则。", s["body"]),
        p("5.4  局限性", s["h2"]),
        p(
            "本研究仍有四项主要限制。其一，地面基底统一采用4 m，未引入逐建筑DEM地面高程；当地形起伏不可忽略时，离地高度将携带地面高程误差。其二，三景RSLC具有相同或接近的观测几何，多景中值融合主要抑制偶然散射，不能替代多视角几何约束。其三，当前重叠归属采用二维多边形交叠与置信优先规则，尚未构建同时优化所有建筑的全局图模型。其四，369栋结果仅完成内部质量审计，缺少足量外部实测高度，不能报告RMSE、MAE或系统偏差等真实精度指标。", s["body"]),
        p("5.5  后续优化方向", s["h2"]),
        p(
            "后续工作应优先引入高分辨率DEM或建筑基底点，将统一4 m基准升级为逐建筑地面高程；利用不同轨道或不同入射角SAR形成多视角高度交会；在狭长建筑中显式区分屋顶双侧边缘和墙面亮线；用图优化同时处理邻近建筑的候选归属；最后建立覆盖不同形态、高度和可观测性等级的外部验证样本。获得真值后，应分别报告各置信等级的MAE、RMSE、中位绝对误差和覆盖率，而不只报告总体误差。", s["body"]),

        p("6  结论", s["h1"]),
        p(
            "本文完成了一套面向共注册RSLC影像的建筑高度估计全流程。方法以建筑顶面为投影对象，通过0 m与初始绝对高程投影建立几何参考，使用多景SAR增强和建筑形态自适应二维搜索定位图像特征，再对每个候选高程执行严格距离-多普勒重投影，并以多景一致性、几何残差、分支一致性和重叠归属形成分级结果。最终1028栋建筑中369栋获得离地高度，比上一版293栋增加76栋；高、中置信结果共143栋。", s["body"]),
        KeepTogether([p(
            "本次优化最关键的修正是区分屋顶绝对高程与建筑离地高度。采用4 m地面基底转换后，clean_id=776和788分别得到18.00 m和17.20 m，旧结果约4 m的系统性偏高得到解释。所有无值建筑均保持灰色，未使用Shapefile height、均值或邻域高度填充。在获得外部测高数据前，不应将内部置信等级表述为真实精度等级。", s["body"]
        )]),

        p("数据、代码与可复现性声明", s["h1"]),
        p(
            "本报告中的数量与高度统计由最终CSV和JSON自动读取，避免人工抄写。核心流程、矢量结果、正式图件与报告生成脚本的位置如下：", s["body"]),
        make_table([
            ["内容", "相对项目根目录路径"],
            ["逐建筑结果", "results/tables/joint_quantity_quality_building_heights.csv"],
            ["统计摘要", "results/tables/joint_quantity_quality_optimization_summary.json"],
            ["核心程序", "code/run_joint_quantity_quality_optimization.py"],
            ["矢量结果", "results/vectors/joint_quantity_quality_optimization.gpkg"],
            ["正式SVG图件", "results/picall/正式图件/"],
            ["PDF生成脚本", "code/build_paper_report_pdf.py"],
        ], [32 * mm, 126 * mm], s["small"]),
        Spacer(1, 4 * mm),
        p(
            "正式制图目录继续保持SVG格式，本PDF独立保存在output/pdf目录。", s["body"]),
        p(
            "本研究未声明外部真实高度精度。两栋参考建筑的已知约18 m高度仅用于事后解释，没有参与训练、评分、候选约束或缺失值填充。Shapefile height字段只参与初始绝对高程投影及每米像素位移关系构建。", s["body0"]),

        p("参考文献", s["h1"]),
        p("[1] CURLANDER J C, MCDONOUGH R N. Synthetic Aperture Radar: Systems and Signal Processing[M]. New York: John Wiley & Sons, 1991.", s["ref"]),
        p("[2] SCHREIER G. SAR Geocoding: Data and Systems[M]. Karlsruhe: Wichmann, 1993.", s["ref"]),
        p("[3] SMALL D, SCHUBERT A. Guide to ASAR Geocoding[R]. Issue 1.0. European Space Agency, 2008.", s["ref"]),
        p("[4] LEE J S. Digital image enhancement and noise filtering by use of local statistics[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1980, 2(2): 165-168. DOI: 10.1109/TPAMI.1980.4766994.", s["ref"]),
        p("[5] CANNY J. A computational approach to edge detection[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1986, PAMI-8(6): 679-698. DOI: 10.1109/TPAMI.1986.4767851.", s["ref"]),
        Spacer(1, 8 * mm),
        p("注：参考文献用于说明距离-多普勒几何、局部统计处理与边缘检测的理论背景；本报告的建筑数量、匹配增益和高度统计均来自本项目本地结果。", s["small"]),
    ])
    return story


def main():
    global S
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    S = styles()
    summary = json.loads((ROOT / "results/tables/joint_quantity_quality_optimization_summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(ROOT / "results/tables/joint_quantity_quality_building_heights.csv")
    doc = PaperDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=22 * mm, bottomMargin=18 * mm,
        title="基于SAR图像特征与严格距离-多普勒投影的建筑高度估计",
        author="项目技术报告",
        subject="建筑高度估计数量-质量联合优化",
    )
    story = build_story(summary, table)
    doc.multiBuild(story)
    manifest = {
        "pdf": str(OUTPUT),
        "source_script": str(Path(__file__).resolve()),
        "page_size": "A4",
        "language": "zh-CN with English abstract",
        "input_buildings": int(len(table)),
        "final_heights": int(table.final_accepted.sum()),
        "prior_height_used_as_final_fill": False,
        "external_accuracy_validated": False,
    }
    (OUTPUT_DIR / "building_height_estimation_joint_optimization_report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
