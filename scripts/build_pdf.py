#!/usr/bin/env python3
"""Generate docs/swarm-harness.pdf — how the harness works + how to call it externally."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle, PageBreak, HRFlowable, ListFlowable, ListItem)

OUT = "/home/sam/Documents/swarm-harness/docs/swarm-harness.pdf"

INK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#2d5b9e")
MUTED = colors.HexColor("#5b6472")
CODEBG = colors.HexColor("#f4f6fb")
CODEBD = colors.HexColor("#d8deea")
BOXBG = colors.HexColor("#eaf0fb")
CONFIRMBG = colors.HexColor("#e9f5ec")

ss = getSampleStyleSheet()
S = {}
S["title"] = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=24, textColor=INK, spaceAfter=4, leading=28)
S["sub"] = ParagraphStyle("s", fontName="Helvetica", fontSize=11, textColor=MUTED,
                          spaceAfter=14, leading=15)
S["h1"] = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, textColor=ACCENT,
                         spaceBefore=13, spaceAfter=6, leading=18)
S["h2"] = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, textColor=INK,
                         spaceBefore=10, spaceAfter=3, leading=14)
S["body"] = ParagraphStyle("b", fontName="Helvetica", fontSize=9.6, textColor=INK,
                           spaceAfter=6, leading=14)
S["small"] = ParagraphStyle("sm", parent=S["body"], fontSize=8.6, textColor=MUTED, leading=12)
S["code"] = ParagraphStyle("c", fontName="Courier", fontSize=8.1, textColor=INK, leading=11.5)
S["codecap"] = ParagraphStyle("cc", parent=S["small"], spaceBefore=1, spaceAfter=8)
S["cell"] = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6, textColor=INK, leading=11.5)
S["cellb"] = ParagraphStyle("cellb", parent=S["cell"], fontName="Helvetica-Bold")
S["boxt"] = ParagraphStyle("bt", fontName="Helvetica-Bold", fontSize=9, textColor=INK,
                           alignment=TA_CENTER, leading=11)
S["boxs"] = ParagraphStyle("bs", fontName="Helvetica", fontSize=7.4, textColor=MUTED,
                           alignment=TA_CENTER, leading=9)

story = []


def P(text, style="body"):
    story.append(Paragraph(text, S[style]))


def space(h=6):
    story.append(Spacer(1, h))


def code(lines, caption=None):
    rows = [[Paragraph(ln.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;") or "&nbsp;", S["code"])]
            for ln in lines]
    t = Table(rows, colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("BOX", (0, 0), (-1, -1), 0.6, CODEBD),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("TOPPADDING", (0, 0), (0, 0), 6), ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
    ]))
    story.append(t)
    if caption:
        P(caption, "codecap")
    else:
        space(8)


def box(title, sub, bg=BOXBG):
    inner = [Paragraph(title, S["boxt"])]
    if sub:
        inner.append(Paragraph(sub, S["boxs"]))
    t = Table([[inner]], colWidths=[33 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.8, ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def arrow():
    return Paragraph("&ndash;&gt;", ParagraphStyle("a", fontName="Helvetica-Bold", fontSize=13,
                                              textColor=ACCENT, alignment=TA_CENTER))


# ---------- page 1 ----------
P("swarm-harness", "title")
P("A multi-agent swarm driven by one local llama.cpp brain, working a task against one target VM.", "sub")
story.append(HRFlowable(width="100%", thickness=1, color=CODEBD, spaceAfter=10))

P("What it is", "h1")
P("The harness runs <b>N cooperating agents</b> as threads. They all talk to a single "
  "OpenAI-compatible endpoint &mdash; <b>llama.cpp</b> serving <b>gpt-oss-20B</b> (MXFP4) on the "
  "RX 6950 XT &mdash; and all act on one target machine over SSH. Agents coordinate on a shared "
  "whiteboard: they gather facts, propose findings, confirm each other's findings, and advance "
  "through phases by consensus. You hand the swarm a <b>task</b> and a <b>target</b>; it works the "
  "task and writes every step to a JSONL event log. Pure Python standard library &mdash; no "
  "third-party dependencies.", "body")

P("Architecture", "h1")
diagram = Table([[
    box("YOUR CALL", "swarm run / Python / config", BOXBG), arrow(),
    box("ORCHESTRATOR", "spawns N agent threads", BOXBG), arrow(),
    box("WHITEBOARD", "state + phases + consensus", CONFIRMBG),
]], colWidths=[36 * mm, 10 * mm, 36 * mm, 10 * mm, 36 * mm])
diagram.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
story.append(diagram)
space(6)
diagram2 = Table([[
    box("AGENT xN", "build ctx -> ask -> act -> log", BOXBG), arrow(),
    box("llama.cpp BRAIN", "gpt-oss-20B @ :8000/v1", BOXBG), arrow(),
    box("SSH EXECUTOR", "persistent session / target VM", BOXBG),
]], colWidths=[36 * mm, 10 * mm, 36 * mm, 10 * mm, 36 * mm])
diagram2.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
story.append(diagram2)
space(4)
P("Each agent loops: build a small context (task + compact state + current-phase whiteboard slice), "
  "ask the brain for one JSON action, perform it, log the event. The brain is shared by all agents; "
  "the whiteboard is the only shared memory.", "small")

P("The agent loop, one turn at a time", "h1")
P("Every turn is exactly one JSON object in the message body &mdash; <b>no native tool-calling</b> "
  "(gpt-oss tool calls over this endpoint are unreliable). The code parses and validates it, and "
  "retries up to 3&times; on malformed output, feeding the bad reply back with a tighter instruction. "
  "After 3 failures the turn degrades to a harmless note so the swarm never wedges; the aggregate "
  "retry rate is logged.", "body")
code([
    '{"action": "run",     "cmd": "systemctl list-units --type=service", "reason": "..."}',
    '{"action": "propose", "proposal": "nginx 1.24 is listening on :80",  "reason": "..."}',
    '{"action": "confirm", "confirm": "P1", "cmd": "curl -sI localhost:80", "reason": "..."}',
    '{"action": "vote",    "phase_done_vote": true, "reason": "..."}',
    '{"action": "note",    "reason": "..."}',
], "The five actions. run/confirm execute on the target; propose/confirm/vote update the whiteboard.")

mech_heading = Paragraph("Four mechanics that make it a swarm, not N loners", S["h1"])
mech = [
    ("Context discipline", "Each turn the model sees only: the task, a compact JSON state dict, and the "
     "current phase's whiteboard slice &mdash; hard-clamped to ~2K tokens. Raw command output never reaches "
     "the model; it is compressed to a few fact lines (<font face='Courier'>[rc=0] ...</font>) in code first."),
    ("Grounded confirmation", "A <font face='Courier'>propose</font> posts an unconfirmed claim. It counts as "
     "confirmed only when a <b>different</b> agent runs its own verification command and that command succeeds. "
     "No agent can confirm its own claim."),
    ("Consensus phases", "Default phases are explore -&gt; verify -&gt; report. An agent "
     "<font face='Courier'>vote</font>s a phase done; when a configurable fraction (default 0.6) of agents agree, "
     "the whiteboard advances. The run ends when the last phase clears or a step cap is hit."),
    ("Persistent SSH", "One multiplexed OpenSSH session per agent (ControlMaster), reused for every command. "
     "Key-based only, never prompts, fails fast on an unreachable host (10&nbsp;s). A <font face='Courier'>local</font> "
     "mode runs commands on your own box for self-tests."),
]
rows = [[Paragraph(t, S["cellb"]), Paragraph(d, S["cell"])] for t, d in mech]
t = Table(rows, colWidths=[38 * mm, 127 * mm])
t.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LINEBELOW", (0, 0), (-1, -2), 0.4, CODEBD),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
]))
story.append(mech_heading)
story.append(KeepTogether(t))

space(4)

# ---------- page 2: calling it externally ----------
P("Calling it for a given task", "h1")
P("You give the swarm two things: a <b>task</b> (free text, injected into every agent's context) and a "
  "<b>target</b> (your lab VM, key-based SSH). There are four ways to invoke it.", "body")

P("1. Command line", "h2")
code([
    "pip install -e .          # once: puts `swarm` on PATH (or use `python -m swarm.cli`)",
    "",
    "swarm run \\",
    '  --task "Inventory running services and report their versions" \\',
    "  --agents 4 \\",
    "  --target 192.168.1.50 --target-user lab --target-key ~/.ssh/id_lab \\",
    "  --model http://localhost:8000/v1 \\",
    "  --phases explore,verify,report \\",
    "  --log run.jsonl",
], "--model is the OpenAI base URL. Exit code 0 = all phases cleared, 1 = did not finish.")

P("2. Config file (reusable targets)", "h2")
code([
    "# swarm.toml",
    'task    = "Inventory running services and report their versions"',
    "agents  = 4",
    'base_url = "http://localhost:8000/v1"',
    'phases  = ["explore", "verify", "report"]',
    "consensus = 0.6",
    "",
    "[target]",
    'host = "192.168.1.50"',
    'user = "lab"',
    'key  = "~/.ssh/id_lab"',
    'mode = "ssh"          # or "local" to run on this box',
    "",
    "#  swarm run --config swarm.toml  --task \"override the task here if you like\"",
], "Precedence: config file &lt; SWARM_* env vars &lt; CLI flags. A JSON file works too.")

P("3. From Python (embed it in your own program)", "h2")
code([
    "from swarm.config import load_config",
    "from swarm.orchestrator import Orchestrator",
    "",
    "cfg = load_config(",
    "    None,                                  # or a path to swarm.toml",
    '    task="Find the web server and confirm its version",',
    "    agents=4,",
    '    base_url="http://localhost:8000/v1",',
    '    target_host="192.168.1.50", target_user="lab",',
    '    target_key="~/.ssh/id_lab", target_mode="ssh",',
    '    log="run.jsonl",',
    ")",
    "summary = Orchestrator(cfg).run()   # blocks until the run ends",
    "",
    "print(summary['all_phases_done'], summary['retry_rate'])",
    "# -> True 0.04   (also: phases_cleared, proposals, confirmed, total_turns, wall_s)",
], "run() returns a summary dict and writes one JSON event per turn to the log.")

P("4. Env vars (containers / CI)", "h2")
code([
    "export SWARM_BASE_URL=http://localhost:8000/v1",
    "export SWARM_TARGET_HOST=192.168.1.50  SWARM_TARGET_USER=lab",
    "export SWARM_TARGET_KEY=~/.ssh/id_lab  SWARM_AGENTS=4",
    'swarm run --task "Audit cron jobs and report anything unusual"',
], None)

P("What you get back: run.jsonl", "h1")
P("One JSON object per line. <font face='Courier'>run_start</font> records the task and config; each turn "
  "records the agent, phase, action, reason, retries and the compressed result; <font face='Courier'>run_end</font> "
  "summarizes.", "body")
code([
    '{"event":"run_start","config":{"agents":4,"task":"Inventory...","phases":[...]}}',
    '{"agent":"agent-1","step":3,"phase":"explore","action":"run",',
    '   "result":{"cmd":"uname -s","rc":0,"digest":"[rc=0] Linux"}, "retries":0}',
    '{"agent":"agent-2","action":"confirm","result":{"target_proposal":"P1",',
    '   "supports":true,"status":"P1 CONFIRMED"}}',
    '{"event":"run_end","all_phases_done":true,"phases_cleared":3,',
    '   "proposals":3,"confirmed":3,"total_turns":21,"retry_rate":0.04}',
], None)

P("Operating notes", "h1")
notes = [
    "<b>Prove the loop with --agents 1 first</b>, then scale up. One agent isolates the loop from concurrency.",
    "<b>Per-agent context = total_ctx / --parallel</b> on the server (32768/8 = 4K). Keep agent prompts &lt;=2K "
    "and raise llama.cpp <font face='Courier'>--parallel</font> for more agents.",
    "<b>SSH must be passwordless</b> (<font face='Courier'>BatchMode=yes</font>). Test by hand first: "
    "<font face='Courier'>ssh -i key user@host uname -a</font>.",
    "<b>If the retry rate is high</b>, suspect harmony reasoning tokens leaking into the JSON body &mdash; "
    "check the server's <font face='Courier'>--reasoning-format auto</font>.",
    "<b>No GPU/VM handy?</b> Run against the mock brain with <font face='Courier'>--local</font> to exercise "
    "every code path on this box with benign commands.",
]
story.append(ListFlowable([ListItem(Paragraph(n, S["body"]), leftIndent=6, value="•") for n in notes],
                          bulletType="bullet", start="•"))

space(3)
P("swarm-harness &bull; gpt-oss-20B on llama.cpp (gfx1030) &bull; generated 2026-11-01", "small")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, "page %d" % doc.page)
    canvas.drawString(18 * mm, 12 * mm, "swarm-harness")
    canvas.restoreState()


import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=16 * mm, bottomMargin=18 * mm, title="swarm-harness",
                      author="swarm-harness")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
doc.build(story)
print("wrote", OUT)
