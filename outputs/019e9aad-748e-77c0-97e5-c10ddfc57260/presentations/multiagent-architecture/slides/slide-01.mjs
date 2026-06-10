export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "MatSci-Agent Multi-Agent Harness",
    left: 76, top: 88, width: 1120, height: 78,
    fontSize: 30, bold: true, color: "#0F172A", face: "Arial",
  });
  ctx.addText(slide, {
    text: "High-level architecture, design reasons, workflow",
    left: 76, top: 162, width: 1040, height: 38,
    fontSize: 20, color: "#2563EB", face: "Arial",
  });
  ctx.addShape(slide, {
    left: 76, top: 228, width: 1128, height: 2,
    fill: "#2563EB",
  });
  ctx.addText(slide, {
    text: "Talk goal",
    left: 76, top: 272, width: 180, height: 28,
    fontSize: 18, bold: true, color: "#0F172A",
  });
  ctx.addText(slide, {
    text: "Explain what scaffold is, why it sits outside main workflow, how agents collaborate, where safety and key config live, what is still missing.",
    left: 76, top: 306, width: 1120, height: 120,
    fontSize: 22, color: "#0F172A",
  });
  ctx.addText(slide, {
    text: "Scope today",
    left: 76, top: 474, width: 180, height: 28,
    fontSize: 18, bold: true, color: "#0F172A",
  });
  ctx.addText(slide, {
    text: "Current branch: multi-agent\nRuntime code: src/matsci_agent/multiagent\nPrompt specs: agent_specs",
    left: 76, top: 510, width: 530, height: 120,
    fontSize: 22, color: "#475569",
  });
  ctx.addShape(slide, {
    left: 764, top: 462, width: 440, height: 150,
    fill: "#DBEAFE",
    line: ctx.line("#93C5FD", 1),
  });
  ctx.addText(slide, {
    text: "Key point:\nThis is scaffolding for eval / critique / repair around existing retrieval app.\nNot replacement for app workflow.",
    left: 792, top: 490, width: 390, height: 102,
    fontSize: 22, color: "#0F172A",
  });
  return slide;
}
