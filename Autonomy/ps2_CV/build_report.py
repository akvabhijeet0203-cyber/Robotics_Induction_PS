"""Build a polished PS2 research report and reproducible demo figures."""
from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageTemplate, PageBreak, Paragraph,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "research.md"
OUTPUT = ROOT / "PS2_tool_sorting_report.pdf"
NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#176B87")
TEAL = colors.HexColor("#18A6A6")
INK = colors.HexColor("#243B53")
PALE = colors.HexColor("#EAF3F6")
MUTED = colors.HexColor("#526777")


def clean_markup(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    return text.replace("&", "&amp;")


def parse_research(text: str, styles):
    story = []
    pagebreak_before = ("## 4. Coordinate", "## Demonstration and observations")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 1.8 * mm))
        elif line.startswith("# "):
            continue
        elif line.startswith("## "):
            if line.startswith(pagebreak_before):
                story.append(PageBreak())
            story.append(Paragraph(clean_markup(line[3:]), styles["SectionPS2"]))
        elif line.startswith("### "):
            story.append(Paragraph(clean_markup(line[4:]), styles["SubsectionPS2"]))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + clean_markup(line[2:]), styles["BodyPS2"]))
        else:
            match = re.match(r"^(\d+)\.\s+(.*)$", line)
            if match:
                story.append(Paragraph(f"<b>{match.group(1)}.</b> {clean_markup(match.group(2))}", styles["BodyPS2"]))
            else:
                story.append(Paragraph(clean_markup(line), styles["BodyPS2"]))
    return story


def page_chrome(canvas, doc):
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 7 * mm, width, 7 * mm, stroke=0, fill=1)
        canvas.setFillColor(TEAL)
        canvas.rect(0, height - 7 * mm, 31 * mm, 7 * mm, stroke=0, fill=1)
    canvas.setStrokeColor(colors.HexColor("#D7E2EA"))
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "IVDC AUTONOMY 2026  /  PS2 COMPUTER VISION")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"{doc.page:02d}")
    canvas.restoreState()


def workflow_figure(path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Acquire", "Preprocess", "Segment", "Contours +\nshape", "Grasp\ncandidate", "Calibrated\ncoordinates"]
    fig, ax = plt.subplots(figsize=(11, 2.25))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, label in enumerate(labels):
        ax.text(index + .5, .52, label, ha="center", va="center", color="#102A43", fontsize=9.5,
                bbox={"boxstyle": "round,pad=0.55", "fc": "#EAF3F6", "ec": "#176B87", "lw": 1.4})
        if index < len(labels) - 1:
            ax.annotate("", xy=(index + .93, .52), xytext=(index + .68, .52),
                        arrowprops={"arrowstyle": "-|>", "color": "#18A6A6", "lw": 1.8})
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    from tool_sorting import create_demo_images, process_image
    output_dir = ROOT / "outputs"
    demo_images = create_demo_images(ROOT / "demo_images")
    for image_path in demo_images:
        process_image(image_path, output_dir)

    flow = output_dir / "workflow.png"
    workflow_figure(flow)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitlePS2", fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=colors.white, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="CoverSubtitlePS2", fontName="Helvetica", fontSize=11, leading=16, textColor=colors.HexColor("#D9EAF2")))
    styles.add(ParagraphStyle(name="SectionPS2", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=BLUE, spaceBefore=4 * mm, spaceAfter=2.2 * mm, keepWithNext=True))
    styles.add(ParagraphStyle(name="SubsectionPS2", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=2.5 * mm, spaceAfter=1 * mm, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyPS2", fontName="Helvetica", fontSize=9.1, leading=12.8, textColor=INK, spaceAfter=1.2 * mm, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="CardPS2", fontName="Helvetica", fontSize=8.6, leading=12, textColor=INK))
    styles.add(ParagraphStyle(name="CaptionPS2", fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=MUTED, alignment=TA_CENTER, spaceAfter=2 * mm))
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=16 * mm, bottomMargin=21 * mm,
                          title="PS2 - Computer Vision for Robotic Object Sorting",
                          author="IVDC Autonomy 2026")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=page_chrome)])

    story = []
    cover = Table([
        [Paragraph("PROBLEM STATEMENT 2<br/>COMPUTER VISION", styles["CoverSubtitlePS2"])],
        [Paragraph("Robotic Object Sorting", styles["CoverTitlePS2"])],
        [Paragraph("Research answers, classical vision pipeline, localization, and experiments", styles["CoverSubtitlePS2"])],
    ], colWidths=[doc.width])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("BOX", (0, 0), (-1, -1), .5, NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
        ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
    ]))
    story.extend([Spacer(1, 5 * mm), cover, Spacer(1, 5 * mm)])
    story.append(Paragraph("Project overview", styles["SectionPS2"]))
    story.append(Paragraph(
        "This submission builds a transparent OpenCV baseline that turns a tabletop image into tool proposals, shape labels, image-space centers, orientation estimates, and candidate grasp points. It explains what the method can show and where it fails, especially under occlusion and without calibration.",
        styles["BodyPS2"]))
    cards = [[
        Paragraph("<b>Classical CV</b><br/>Blur, HSV/GrabCut, Canny, morphology, contours, and silhouette matching.", styles["CardPS2"]),
        Paragraph("<b>Visible output</b><br/>Intermediate masks, annotated candidates, JSON/CSV coordinates, and workflow diagram.", styles["CardPS2"]),
        Paragraph("<b>Scope boundary</b><br/>Generated illustrations only; no robot commands or claimed dataset accuracy.", styles["CardPS2"]),
    ]]
    card_table = Table(cards, colWidths=[doc.width / 3] * 3)
    card_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), .6, colors.HexColor("#C4D6E0")),
        ("INNERGRID", (0, 0), (-1, -1), .5, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm), ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
    ]))
    story.extend([card_table, Paragraph("End-to-end workflow", styles["SectionPS2"]),
                  Image(str(flow), width=doc.width, height=39 * mm),
                  Paragraph("The Canny edge map is shown as a diagnostic. Contours are extracted from cleaned foreground masks so their regions remain filled.", styles["CaptionPS2"]),
                  PageBreak(), Paragraph("Demonstration scenes", styles["SectionPS2"]),
                  Paragraph("Both inputs below are generated illustrations, not photographs from the linked dataset. They make the segmentation and overlap limitations reproducible.", styles["BodyPS2"])])
    for image_path in demo_images:
        analysis_path = output_dir / f"{image_path.stem}_analysis.png"
        story.append(Image(str(analysis_path), width=doc.width, height=doc.width / 2))
        caption = "Separated objects: three candidate silhouettes." if "separated" in image_path.stem else "Occlusion case: visible fragments may be split or assigned the wrong shape label."
        story.append(Paragraph(f"{image_path.stem.replace('_', ' ')} - {caption}", styles["CaptionPS2"]))
        story.append(Spacer(1, 2 * mm))
    story.append(PageBreak())
    story.extend(parse_research(RESEARCH.read_text(encoding="utf-8"), styles))
    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
