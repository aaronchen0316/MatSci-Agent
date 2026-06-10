export async function slide02(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "Why Add Multi-Agent Layer",
    left: 76, top: 64, width: 1000, height: 52,
    fontSize: 28, bold: true, color: "#0F172A",
  });
  const bullets = [
    "Main app already good at deterministic retrieval execution.",
    "Weak spot is outer loop: evaluate retrieval quality, diagnose failures, repair code, re-check result.",
    "Different tasks need different reasoning modes: testing, criticism, debugging, verification.",
    "Manager-style orchestration keeps those roles separate instead of mixing them into one long prompt.",
  ];
  let top = 156;
  for (const bullet of bullets) {
    ctx.addShape(slide, { left: 84, top: top + 10, width: 10, height: 10, fill: "#2563EB" });
    ctx.addText(slide, {
      text: bullet,
      left: 110, top, width: 1060, height: 58,
      fontSize: 24, color: "#0F172A",
    });
    top += 88;
  }
  ctx.addShape(slide, {
    left: 76, top: 546, width: 1128, height: 112,
    fill: "#EFF6FF",
    line: ctx.line("#BFDBFE", 1),
  });
  ctx.addText(slide, {
    text: "Design reason: keep shortlist logic deterministic inside app. Put fuzzy reasoning outside app, where it can inspect, grade, and repair without changing product contract.",
    left: 96, top: 574, width: 1080, height: 72,
    fontSize: 22, color: "#1E3A8A",
  });
  return slide;
}
