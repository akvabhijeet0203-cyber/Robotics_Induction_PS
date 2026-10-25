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
INK = colors.HexColor("#202020")
MUTED = colors.HexColor("#555555")


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
    canvas.setStrokeColor(colors.HexColor("#B8B8B8"))
    if doc.page > 1:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, height - 11 * mm, "IVDC AUTONOMY 2026  |  PROBLEM STATEMENT 2")
        canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
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
        ax.text(index + .5, .52, label, ha="center", va="center", color="#202020", fontsize=9.5,
                bbox={"boxstyle": "square,pad=0.5", "fc": "white", "ec": "#555555", "lw": 1.0})
        if index < len(labels) - 1:
            ax.annotate("", xy=(index + .93, .52), xytext=(index + .68, .52),
                        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.0})
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
    styles.add(ParagraphStyle(name="CoverTitlePS2", fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=INK, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="CoverSubtitlePS2", fontName="Helvetica", fontSize=10, leading=14, textColor=MUTED))
    styles.add(ParagraphStyle(name="SectionPS2", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=4 * mm, spaceAfter=2.2 * mm, keepWithNext=True))
    styles.add(ParagraphStyle(name="SubsectionPS2", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=INK, spaceBefore=2.5 * mm, spaceAfter=1 * mm, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyPS2", fontName="Helvetica", fontSize=9.1, leading=12.8, textColor=INK, spaceAfter=1.2 * mm, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="CaptionPS2", fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=MUTED, alignment=TA_CENTER, spaceAfter=2 * mm))
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                          topMargin=16 * mm, bottomMargin=21 * mm,
                          title="PS2 - Computer Vision for Robotic Object Sorting",
                          author="IVDC Autonomy 2026")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="report", frames=frame, onPage=page_chrome)])

    story = []
    story.extend([Spacer(1, 14 * mm),
                  Paragraph("IVDC AUTONOMY 2026  |  PROBLEM STATEMENT 2", styles["CoverSubtitlePS2"]),
                  Spacer(1, 3 * mm), Paragraph("Computer Vision for<br/>Robotic Object Sorting", styles["CoverTitlePS2"]),
                  Spacer(1, 3 * mm), Paragraph("Research, implementation and evaluation", styles["CoverSubtitlePS2"]),
                  Spacer(1, 5 * mm)])
    rule = Table([[""]], colWidths=[doc.width], rowHeights=[.5 * mm])
    rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#777777"))]))
    story.extend([rule, Spacer(1, 5 * mm)])
    story.append(Paragraph("Project overview", styles["SectionPS2"]))
    story.append(Paragraph(
        "This submission builds a transparent OpenCV baseline that turns a tabletop image into tool proposals, shape labels, image-space centers, orientation estimates, and candidate grasp points. It explains what the method can show and where it fails, especially under occlusion and without calibration.",
        styles["BodyPS2"]))
    story.extend([Paragraph("End-to-end workflow", styles["SectionPS2"]),
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
