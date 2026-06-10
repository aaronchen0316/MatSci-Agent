export async function slide06(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = "#F8FAFC";

  ctx.addText(slide, {
    text: "Architecture Chart",
    left: 76, top: 52, width: 900, height: 50,
    fontSize: 28, bold: true, color: "#0F172A",
  });

  const box = (x, y, w, h, title, body, fill = "#FFFFFF") => {
    ctx.addShape(slide, { left: x, top: y, width: w, height: h, fill, line: ctx.line("#94A3B8", 1) });
    ctx.addText(slide, {
      text: title, left: x + 16, top: y + 14, width: w - 32, height: 24,
      fontSize: 19, bold: true, color: "#1D4ED8",
    });
    ctx.addText(slide, {
      text: body, left: x + 16, top: y + 44, width: w - 32, height: h - 56,
      fontSize: 17, color: "#0F172A",
    });
  };
  const line = (x, y, w, h) => ctx.addShape(slide, { left: x, top: y, width: w, height: h, fill: "#2563EB" });

  box(470, 110, 340, 88, "Controller Agent", "Uses specialist tools in manager style.\nOwns final answer.", "#DBEAFE");
  box(80, 252, 250, 108, "Retrieval Tester", "Checks traces, fixtures,\nfuture eval commands.");
  box(380, 252, 250, 108, "Query Critic", "Maps failure -> root cause -> module.");
  box(680, 252, 250, 108, "Codex Debugger", "Creates repair branch / commit /\nPR flow when enabled.");
  box(980, 252, 220, 108, "Final Verifier", "Reviews repair claim.\nMay request tester refresh.");

  box(120, 500, 430, 128, "Shared tool layer", "read_context_snapshot\nread_repo_file\nlist_repo_files\nrun_readonly_repo_command", "#FFFFFF");
  box(706, 500, 414, 128, "Mutation tools", "create_branch_worktree\nread_worktree_diff\ncommit_worktree_changes\ncreate_pull_request", "#FFFFFF");

  box(450, 462, 360, 164, "Existing app backbone", "src/matsci_agent/workflow/graph.py\nsrc/matsci_agent/tools/policy_filter.py\nsrc/matsci_agent/tools/mp_retriever.py\nsrc/matsci_agent/api/main.py", "#F8FAFC");

  line(640, 198, 4, 42);
  line(205, 224, 4, 28); line(505, 224, 4, 28); line(805, 224, 4, 28); line(1085, 224, 4, 28);
  line(200, 360, 4, 130); line(790, 360, 4, 130);
  line(630, 420, 4, 38);
  line(330, 560, 120, 4); line(810, 560, 120, 4);

  ctx.addText(slide, {
    text: "Today: scaffold exists. Full eval runner + real code-edit loop still thin.",
    left: 76, top: 660, width: 1120, height: 26,
    fontSize: 16, color: "#475569",
  });
  return slide;
}
