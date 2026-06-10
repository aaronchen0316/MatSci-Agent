export async function slide04(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "Where Things Live",
    left: 76, top: 64, width: 900, height: 52,
    fontSize: 28, bold: true, color: "#0F172A",
  });

  const rows = [
    ["Prompt specs", "agent_specs/", "Role instructions only. No runtime logic."],
    ["Typed contracts", "src/matsci_agent/multiagent/schemas.py", "Input / output schema for each agent."],
    ["Tool gating", "src/matsci_agent/multiagent/tools.py", "Read-only tools, git tools, PR tools."],
    ["SDK config", "src/matsci_agent/multiagent/sdk.py", "Shared client, model, tracing setting."],
    ["Agent construction", "src/matsci_agent/multiagent/factory.py", "Build controller + specialist agents."],
    ["Top orchestrator", "src/matsci_agent/multiagent/orchestrator.py", "Wrap specialist calls for controller."],
    ["CLI entry", "src/matsci_agent/multiagent/cli.py", "Plan or run harness."],
  ];

  let y = 144;
  for (const [a, b, c] of rows) {
    ctx.addShape(slide, {
      left: 76, top: y, width: 1128, height: 58,
      fill: y % 116 === 28 ? "#FFFFFF" : "#F1F5F9",
      line: ctx.line("#E2E8F0", 1),
    });
    ctx.addText(slide, {
      text: a, left: 92, top: y + 14, width: 220, height: 28,
      fontSize: 18, bold: true, color: "#0F172A",
    });
    ctx.addText(slide, {
      text: b, left: 332, top: y + 14, width: 380, height: 28,
      fontSize: 16, color: "#1D4ED8", face: "Courier New",
    });
    ctx.addText(slide, {
      text: c, left: 732, top: y + 14, width: 452, height: 28,
      fontSize: 18, color: "#475569",
    });
    y += 64;
  }
  return slide;
}
