export async function slide09(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "What Scaffold Can And Cannot Do Today",
    left: 76, top: 64, width: 1040, height: 52,
    fontSize: 28, bold: true, color: "#0F172A",
  });

  ctx.addShape(slide, { left: 76, top: 150, width: 520, height: 460, fill: "#FFFFFF", line: ctx.line("#CBD5E1", 1) });
  ctx.addText(slide, {
    text: "Already present",
    left: 98, top: 176, width: 220, height: 30,
    fontSize: 22, bold: true, color: "#166534",
  });
  ctx.addText(slide, {
    text: "- prompts/specs for 5 agents\n- typed I/O schemas\n- shared OpenAI-compatible client config\n- repo reading tools\n- git worktree / commit / PR scaffolding\n- manager-style controller wrapper\n- CLI entry for plan/run",
    left: 98, top: 222, width: 458, height: 280,
    fontSize: 22, color: "#0F172A",
  });

  ctx.addShape(slide, { left: 684, top: 150, width: 520, height: 460, fill: "#FFFFFF", line: ctx.line("#CBD5E1", 1) });
  ctx.addText(slide, {
    text: "Still missing",
    left: 706, top: 176, width: 220, height: 30,
    fontSize: 22, bold: true, color: "#92400E",
  });
  ctx.addText(slide, {
    text: "- richer formatted prompts with examples\n- real retrieval eval runner\n- tighter controller retry state machine\n- real code-edit execution path for debugger\n- test execution wrapper\n- live trace integration and result artifacts",
    left: 706, top: 222, width: 458, height: 280,
    fontSize: 22, color: "#0F172A",
  });
  return slide;
}
