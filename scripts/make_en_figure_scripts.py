"""Create English versions of the report figure scripts (scripts/en/*.py) writing into report/pdf_en/."""
import os, re
T = {
 '"100%\\n(约 354 个)"': '"100%\\n(~354)"', '"25%\\n(约 89 个)"': '"25%\\n(~89)"', '"50%\\n(约 177 个)"': '"50%\\n(~177)"',
 '"2018 → 2022 水情"': '"2018 → 2022 water status"', '"2018 → 2022 水情（自监督集成模型）"': '"2018 → 2022 water status (SSL ensemble)"',
 '"2022 年已干"': '"Dry by 2022"', '"三年都有水"': '"Wet all three years"', '"间歇"': '"Intermittent"', '"新蓄水/重新蓄水"': '"New / refilled"',
 '"2 个全局视图\\n224 px，部分 patch\\n被遮挡（iBOT）"': '"2 global views\\n224 px, some patches\\nmasked (iBOT)"',
 '"6 个局部视图\\n96 px"': '"6 local views\\n96 px"',
 '"DINOv3-sat 冻结 + 线性"': '"DINOv3-sat frozen + linear"', '"DINOv3-sat 微调"': '"DINOv3-sat fine-tuned"',
 '"DINOv3-sat 特征 PCA"': '"DINOv3-sat feature PCA"', '"DINOv3-sat 稠密检测 + ViT 分类"': '"DINOv3-sat dense detector + ViT classifier"',
 'f"中位数 {tk.diam_m.median():.1f} m"': 'f"median {tk.diam_m.median():.1f} m"',
 'f"全县圆形水槽/储水罐（{len(tk):,} 个）"': 'f"Round tanks / troughs, county-wide ({len(tk):,})"',
 'f"全县圆形水槽/储水罐（{len(tk):,} 个，自监督模型）"': 'f"Round tanks / troughs, county-wide ({len(tk):,}, SSL model)"',
 '"Google 卫星图\\nz19 ≈0.25 m\\n全县 52 万瓦片\\n+ TWDB 井位"': '"Google satellite\\nz19 ≈0.25 m\\n520 k county tiles\\n+ TWDB wells"',
 '"Hamilton County 县界"': '"Hamilton County boundary"', '"NAIP 真彩色"': '"NAIP true colour"',
 '"NAIP 航空影像\\n2018 / 2020 / 2022\\n0.6 m · RGB+近红外\\n77 幅 × 3 年"': '"NAIP aerial imagery\\n2018 / 2020 / 2022\\n0.6 m · RGB+NIR\\n77 tiles × 3 years"',
 '"NAIP 近红外假彩色"': '"NAIP NIR false colour"', '"近红外假彩色"': '"NIR false colour"',
 '"ResNet18（ImageNet 监督）"': '"ResNet18 (ImageNet supervised)"', '"TWDB 牲畜用水井（292 口）"': '"TWDB livestock wells (292)"',
 '"U-Net\\nEfficientNet-B4\\n4 波段输入"': '"U-Net\\nEfficientNet-B4\\n4-band input"',
 '"U-Net v1（4 波段）"': '"U-Net v1 (4-band)"', '"U-Net v2（4 波段，最终）"': '"U-Net v2 (4-band, final)"', '"U-Net v2（仅 RGB）"': '"U-Net v2 (RGB only)"',
 '"YOLO11s\\n候选检测\\n(召回优先)"': '"YOLO11s\\ncandidate detection\\n(recall first)"',
 '"YOLO11s + ResNet18（原方案）"': '"YOLO11s + ResNet18 (original)"',
 '"z20 ≈0.13 m 分类\\nResNet18 · 4 类\\n水槽 / 喂料架\\n草捆 / 其他"': '"z20 ≈0.13 m classifier\\nResNet18 · 4 classes\\ntank / hay ring\\nbale / other"',
 '"两阶段\\n(最终)"': '"Two-stage\\n(final)"', '"单阶段 YOLO\\n(高置信度)"': '"Single-stage YOLO\\n(high confidence)"',
 '"严格"': '"strict"', '"宽松"': '"lenient"',
 '"人工标注\\n203 个水槽框\\n2 轮主动学习"': '"Manual labels\\n203 tank boxes\\n2 active-learning rounds"',
 '"估计直径 (m)"': '"Estimated diameter (m)"', '"数量"': '"Count"',
 '"储水设施"': '"Water facility"', '"其他"': '"Other"', '"圆草捆"': '"Round bale"', '"草捆喂料架"': '"Hay-ring feeder"',
 '"储水设施 F1（5 折交叉验证）"': '"Water-facility F1 (5-fold CV)"',
 '"全县 YOLO 候选的 z20 分类结果（置信度 ≥0.2）"': '"z20 classes of county YOLO candidates (conf ≥ 0.2)"',
 '"全县有水池塘数与水面面积"': '"Wet ponds and open water, county-wide"', '"有水池塘数"': '"Ponds holding water"',
 '"有水池塘比例（按池塘大小）"': '"Share of ponds holding water (by size)"', '"池塘开放水面 (km²)"': '"Pond open water (km²)"',
 '"召回（18 个参照储水设施）"': '"Recall (18 reference facilities)"', '"圆形水槽分支（Google 卫星图）"': '"Round-tank branch (Google satellite)"',
 '"池塘分支（公开数据）"': '"Pond branch (public data)"',
 '"学生网络 ViT-L/16\\n3 亿参数\\nDINOv3-sat 初始化\\n(NIR 滤波器零初始化)"': '"Student ViT-L/16\\n300 M params\\nDINOv3-sat init\\n(NIR filters zero-init)"',
 '"弱标签\\nNWI 池塘位置\\n+ 近红外光谱水体\\n道路/建筑 = 负样本"': '"Weak labels\\nNWI pond locations\\n+ NIR spectral water\\nroads/roofs = negatives"',
 '"弱标签（蓝=水体，黄=忽略）"': '"Weak label (blue = water, yellow = ignore)"',
 '"推理 + 矢量化\\n辐射匹配\\n(2018/2020)"': '"Inference +\\nvectorisation\\nradiometric match\\n(2018/2020)"',
 '"教师网络\\n= 学生权重 EMA\\n不反向传播"': '"Teacher\\n= EMA of student\\nno backprop"',
 '"本县未标注影像\\nNAIP 3 年 RGB+NIR\\n或 Google 0.25 m"': '"Unlabeled county\\nimagery: NAIP 3 yrs\\nRGB+NIR or\\nGoogle 0.25 m"',
 '"水槽检测模型（YOLO11s）"': '"Tank detector (YOLO11s)"', '"水槽清单\\n1,849 个\\n(全县)"': '"Tank inventory\\n1,849\\n(county)"',
 '"水槽直径分布"': '"Tank diameter distribution"', '"水槽类型分类：标注效率"': '"Tank-type classifier: label efficiency"',
 '"池塘分割模型"': '"Pond segmentation"', '"池塘清单\\n逐年水面面积\\n有水/干涸"': '"Pond inventory\\nwater area per year\\nwet / dry"',
 '"测试区（16 幅，约 650 km²）"': '"Test area (16 tiles, ~650 km²)"', '"验证区（8 幅）"': '"Validation (8 tiles)"', '"训练区（53 幅）"': '"Training (53 tiles)"',
 '"测试区储水设施精度"': '"Test-area water-facility precision"', '"测试区按池塘大小的召回"': '"Test-area recall by pond size"',
 '"测试区水槽检测：精度–召回"': '"Test-area tank detection: precision–recall"', '"目标召回（对弱标签）"': '"Object recall (vs weak labels)"',
 '"精度（严格–宽松区间中点）"': '"Precision (mid-point of strict–lenient)"', '"累计 epoch（三轮依次接续训练）"': '"Cumulative epoch (3 successive rounds)"',
 '"训练标注量（每折）"': '"Training labels (per fold)"',
 '"训练目标\\n\\nDINO：[CLS] 跨视图自蒸馏\\n（16,384 个原型）\\n\\niBOT：预测被遮挡 patch\\n的教师特征\\n\\nGram anchoring：patch 间\\n相似结构锚定原模型\\n\\nKoLeo：特征均匀分布"':
   '"Objectives\\n\\nDINO: cross-view [CLS]\\nself-distillation\\n(16,384 prototypes)\\n\\niBOT: predict teacher\\nfeatures of masked patches\\n\\nGram anchoring: keep\\npatch-similarity structure\\nof the original model\\n\\nKoLeo: spread features"',
 '"领域自监督 2×"': '"Domain SSL 2×"', '"领域自监督（Google）稠密检测 + 分类"': '"Domain SSL (Google) dense detector + classifier"',
 '"领域自监督 (RGB+NIR) 特征 PCA"': '"Domain SSL (RGB+NIR) feature PCA"', '"领域自监督 冻结 + 线性"': '"Domain SSL frozen + linear"',
 '"领域自监督 微调"': '"Domain SSL fine-tuned"', '"领域自适应\\n的骨干网络\\n→ 下游微调"': '"Domain-adapted\\nbackbone\\n→ fine-tuning"',
 '"验证集 IoU"': '"Validation IoU"', '"验证集 mAP@0.5"': '"Validation mAP@0.5"',
 'label=f"{lab_}（{len(s):,}）"': 'label=f"{lab_} ({len(s):,})"', 'label=f"{lab}（{len(s):,}）"': 'label=f"{lab} ({len(s):,})"',
 '"U-Net v2\\n(ImageNet CNN)"': '"U-Net v2\\n(ImageNet CNN)"',
 'OUT = "report/pdf"': 'OUT = "report/pdf_en"',
}
os.makedirs("scripts/en", exist_ok=True)
for f in ["24_pdf_figures.py", "25_ssl_report_figures.py", "28_ssl_county_figures.py", "ssl_viz_pca.py"]:
    s = open(f"scripts/{f}").read()
    for a, b in T.items(): s = s.replace(a, b)
    s = s.replace("sys.path.insert(0, os.path.dirname(__file__))", "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))")
    s = s.replace('json.dump(S, open("outputs/summary_hamilton_ssl.json", "w"), indent=1)', 'pass')
    left = re.findall(r'"[^"\n]*[一-鿿][^"\n]*"', s)
    open(f"scripts/en/{f}", "w").write(s); print(f, "untranslated:", left)
