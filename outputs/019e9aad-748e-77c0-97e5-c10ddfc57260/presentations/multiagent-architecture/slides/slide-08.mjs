export async function slide08(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "Safety, Sandbox, Keys",
    left: 76, top: 64, width: 860, height: 52,
    fontSize: 28, bold: true, color: "#0F172A",
  });

  ctx.addShape(slide, { left: 76, top: 148, width: 540, height: 440, fill: "#FFFFFF", line: ctx.line("#CBD5E1", 1) });
  ctx.addText(slide, {
    text: "Safety controls in code",
    left: 98, top: 174, width: 320, height: 28,
    fontSize: 22, bold: true, color: "#2563EB",
  });
  ctx.addText(slide, {
    text: "- settings.py gates live MP / git write / PR\n- tools.py separates read-only vs mutation tools\n- prompts add soft rules\n- host sandbox still strongest hard boundary",
    left: 98, top: 220, width: 470, height: 210,
    fontSize: 22, color: "#0F172A",
  });

  ctx.addShape(slide, { left: 664, top: 148, width: 540, height: 440, fill: "#FFFFFF", line: ctx.line("#CBD5E1", 1) });
  ctx.addText(slide, {
    text: "API key model",
    left: 686, top: 174, width: 240, height: 28,
    fontSize: 22, bold: true, color: "#2563EB",
  });
  ctx.addText(slide, {
    text: "- one shared client for all sub-agents\n- config from MULTIAGENT_API_KEY, MULTIAGENT_BASE_URL, MULTIAGENT_MODEL\n- OpenAI key works\n- OpenAI-compatible proxy key can work if syntax truly compatible\n- one key per sub-agent not needed",
    left: 686, top: 220, width: 470, height: 230,
    fontSize: 22, color: "#0F172A",
  });

  ctx.addText(slide, {
    text: "Current default mode: conservative. No live MP, no git writes, no PR creation unless flags enabled.",
    left: 76, top: 626, width: 1120, height: 30,
    fontSize: 20, color: "#92400E",
  });
  return slide;
}
