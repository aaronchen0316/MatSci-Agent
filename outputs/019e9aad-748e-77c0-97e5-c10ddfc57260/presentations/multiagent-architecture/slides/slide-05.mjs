export async function slide05(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "Sub-Agent Responsibilities",
    left: 76, top: 64, width: 980, height: 52,
    fontSize: 28, bold: true, color: "#0F172A",
  });

  const cards = [
    ["Controller", "Owns loop. Decides next specialist call. Produces final summary."],
    ["Retrieval Tester", "Grades retrieval quality. Labels failure stage. Collects evidence."],
    ["Materials Query Critic", "Maps failure to likely root cause and owning module."],
    ["Codex Debugger", "Prepares repair path. Uses worktree / commit / PR tools when enabled."],
    ["Final Verifier", "Reviews claimed fix. Decides pass, fail, or tester refresh."],
  ];
  let top = 140;
  for (const [title, body] of cards) {
    ctx.addShape(slide, {
      left: 76, top, width: 1128, height: 88,
      fill: "#FFFFFF", line: ctx.line("#CBD5E1", 1),
    });
    ctx.addText(slide, {
      text: title, left: 96, top: top + 18, width: 250, height: 28,
      fontSize: 20, bold: true, color: "#2563EB",
    });
    ctx.addText(slide, {
      text: body, left: 320, top: top + 18, width: 850, height: 42,
      fontSize: 20, color: "#0F172A",
    });
    top += 100;
  }
  return slide;
}
