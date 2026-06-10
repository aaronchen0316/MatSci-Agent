export async function slide07(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "How Collaboration Works",
    left: 76, top: 64, width: 920, height: 52,
    fontSize: 28, bold: true, color: "#0F172A",
  });
  const steps = [
    "1. Controller receives objective.",
    "2. Controller calls Retrieval Tester tool.",
    "3. Tester returns staged failure report or pass.",
    "4. If fail -> Controller calls Materials Query Critic.",
    "5. Critic returns root cause and module targets.",
    "6. Controller calls Codex Debugger with tester + critic reports.",
    "7. Debugger may create worktree / commit / PR when enabled.",
    "8. Controller calls Final Verifier.",
    "9. Verifier returns pass / fail / needs tester refresh.",
    "10. Controller stops or repeats.",
  ];
  let y = 144;
  for (const step of steps) {
    ctx.addText(slide, {
      text: step,
      left: 92, top: y, width: 1088, height: 34,
      fontSize: 22, color: "#0F172A",
    });
    y += 48;
  }
  ctx.addShape(slide, {
    left: 76, top: 632, width: 1128, height: 56,
    fill: "#EFF6FF", line: ctx.line("#BFDBFE", 1),
  });
  ctx.addText(slide, {
    text: "Important: collaboration is manager-style tool calling, not direct agent handoff chain.",
    left: 98, top: 648, width: 1080, height: 24,
    fontSize: 20, color: "#1E3A8A",
  });
  return slide;
}
