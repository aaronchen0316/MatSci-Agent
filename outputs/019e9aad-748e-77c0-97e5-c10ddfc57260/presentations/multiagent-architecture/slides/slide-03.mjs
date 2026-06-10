export async function slide03(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "Two Layers: App Workflow vs Repair Harness",
    left: 76, top: 64, width: 1040, height: 52,
    fontSize: 28, bold: true, color: "#0F172A",
  });

  ctx.addShape(slide, {
    left: 76, top: 152, width: 520, height: 460,
    fill: "#FFFFFF", line: ctx.line("#CBD5E1", 1),
  });
  ctx.addText(slide, {
    text: "Existing product workflow",
    left: 100, top: 178, width: 300, height: 30,
    fontSize: 22, bold: true, color: "#2563EB",
  });
  ctx.addText(slide, {
    text: "Location:\nsrc/matsci_agent/workflow/graph.py\n\nJob:\n- parse intent\n- guardrail task\n- expand search space\n- retrieve MP candidates\n- policy filter\n- rank and report\n\nProperty:\nDeterministic shortlist backbone",
    left: 100, top: 224, width: 454, height: 330,
    fontSize: 22, color: "#0F172A",
  });

  ctx.addShape(slide, {
    left: 684, top: 152, width: 520, height: 460,
    fill: "#FFFFFF", line: ctx.line("#CBD5E1", 1),
  });
  ctx.addText(slide, {
    text: "New multi-agent harness",
    left: 708, top: 178, width: 320, height: 30,
    fontSize: 22, bold: true, color: "#2563EB",
  });
  ctx.addText(slide, {
    text: "Location:\nsrc/matsci_agent/multiagent\nagent_specs\n\nJob:\n- inspect repo and traces\n- evaluate retrieval quality\n- diagnose root cause\n- prepare repair branch / commit / PR flow\n- verify claimed fix\n\nProperty:\nSupervision and repair shell around app",
    left: 708, top: 224, width: 454, height: 330,
    fontSize: 22, color: "#0F172A",
  });
  return slide;
}
