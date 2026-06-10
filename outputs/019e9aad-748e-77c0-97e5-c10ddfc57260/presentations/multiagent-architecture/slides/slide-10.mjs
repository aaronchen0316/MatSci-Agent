export async function slide10(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "Recommended Next Steps",
    left: 76, top: 64, width: 980, height: 52,
    fontSize: 28, bold: true, color: "#0F172A",
  });
  const items = [
    "1. Upgrade prompts into strict templates with few-shot examples.",
    "2. Add explicit offline retrieval eval runner before any live MP path.",
    "3. Add real debugger edit flow with bounded file / branch policy.",
    "4. Add verifier rubric tied to tester failure stages.",
    "5. Only then enable live MP and PR mode.",
  ];
  let y = 164;
  for (const item of items) {
    ctx.addText(slide, {
      text: item,
      left: 92, top: y, width: 1080, height: 42,
      fontSize: 24, color: "#0F172A",
    });
    y += 74;
  }
  ctx.addShape(slide, {
    left: 76, top: 560, width: 1128, height: 98,
    fill: "#DBEAFE", line: ctx.line("#93C5FD", 1),
  });
  ctx.addText(slide, {
    text: "Bottom line: current architecture direction is sound. Current implementation is scaffold, not full autonomous repair system yet.",
    left: 96, top: 592, width: 1088, height: 42,
    fontSize: 23, color: "#1E3A8A",
  });
  return slide;
}
