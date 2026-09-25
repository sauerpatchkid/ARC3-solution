"""Build docs/reports/Goose_Upgrade_Options.docx: the ranked list of candidate
upgrades to the novelty label and the return map, in plain language.

    uv run --with python-docx python docs/reports/make_upgrade_doc.py docs/reports/Goose_Upgrade_Options.docx
"""
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

OUT = sys.argv[1]
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x59, 0x59, 0x59)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(0.9)
sec.top_margin = sec.bottom_margin = Inches(0.8)
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.08


def heading(text, size=13, before=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    r.font.color.rgb = ACCENT
    return p


def para(text, lead=None, after=None, size=None, italic=False):
    p = doc.add_paragraph()
    if lead:
        p.add_run(lead + " ").bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        for run in p.runs:
            run.font.size = Pt(size)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullet(text, lead=None, style="List Bullet", size=None):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(2)
    if lead:
        p.add_run(lead + " ").bold = True
    p.add_run(text)
    if size:
        for run in p.runs:
            run.font.size = Pt(size)
    return p


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def table(rows, widths, header=True, size=9.5):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = widths[j]
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            run = p.add_run(val)
            run.font.size = Pt(size)
            if header and i == 0:
                run.bold = True
                shade(c, "DCE6F1")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def candidate(rank, title, arm, what, why, pros, cons, test):
    heading(f"{rank}. {title}", size=12, before=10)
    para(what, "What it is.", after=3)
    para("", "Why it should help.", after=1)
    for lead, text in why:
        bullet(text, lead)
    p = para("", "Pros.", after=1)
    for text in pros:
        bullet(text)
    para("", "Cons and risks.", after=1)
    for text in cons:
        bullet(text)
    para(test, f"How the screen tests it (arm “{arm}”).", after=4)


# ---------------------------------------------------------------------------------------------
t = doc.add_paragraph()
t.paragraph_format.space_after = Pt(0)
r = t.add_run("Upgrading Goose: Options and a Screening Plan")
r.bold = True
r.font.size = Pt(18)
r.font.color.rgb = ACCENT
st = doc.add_paragraph()
st.paragraph_format.space_after = Pt(6)
r = st.add_run("ARC-AGI-3 Capstone, Team B  |  Next steps after the novelty question and the map  |  September 2026")
r.font.size = Pt(9.5)
r.font.color.rgb = GREY

heading("Summary", before=4)
para("We have two changes to Goose so far. The new question (“did this move reach a screen I haven’t seen "
     "this level?”) was adopted: 79 levels against 54 on all 25 games. The map was not adopted: it helped the "
     "games Goose was stuck on and hurt the games Goose already solved. This document lists nine ways to improve "
     "one or the other, most promising first, with the evidence for each, and describes a single screening "
     "sweep that tries all nine briefly and ranks them. The screen is built and tested but has not been run.")
para("The ranking below is our best judgement before any results. The screen decides. Only the best one to "
     "three ideas will get a proper test.", italic=True)

heading("What our own data points to")
para("Before searching for ideas we re-analysed the runs we already have. Five findings shaped the list:", after=2)
table([
    ("Finding", "What we measured", "Ideas it points to"),
    ("Progress bars are still being counted as “new screens”",
     "The current decoration detector masks nothing on 12 of 14 games checked. A stronger detector finds a "
     "bar on 12 of the 25 games. On ft09 the missed bar inflates “new screens” by 1.4 times.",
     "1 (better bar detection)"),
    ("The map’s walk back drifts on some games",
     "On dc22 and g50t the missed bar shows how many moves were used. A shorter route back shows a different "
     "bar, so the screen looks new and the map gives up.", "1"),
    ("The map helps stuck games and hurts solved ones",
     "25-game test: tu93, vc33, bp35, lf52 up; ar25, tr87 down. On solved click games the map chose about "
     "half of all moves, mostly walking back.", "2, 4, 5"),
    ("Goose repeats clicks it already knows do nothing",
     "Share of clicks that repeat a known dead click: su15 54%, vc33 51%, tn36 47%, lf52 23%, s5i5 21%.", "3"),
    ("Levels restart constantly", "A game over every 20 to 200 moves on most games.", "2, 6"),
], (Inches(1.9), Inches(3.4), Inches(1.4)))

# ---------------------------------------------------------------------------------------------
heading("The options, most promising first")

candidate(
    1, "Catch the progress bars Goose still misses", "bars (and map_bars)",
    "Goose decides whether a screen is new after blanking out decorations. The current rule only spots a "
    "progress bar when the bar is the only thing that changed, which almost never happens, because the bar "
    "usually moves in the same move as a real change. The upgrade looks at each small patch of change on its "
    "own, so a bar ticking at the edge of the screen is caught even when something else changed in the middle. "
    "Two safety rules: only cells near the screen edge count (where every real bar we found sits), and once a "
    "cell is marked as decoration it stays marked.",
    [("Our data:", "12 of 25 games have a bar the current rule misses; on ft09 it inflates the count of new "
                   "screens by 1.4 times, so Goose is rewarded for the bar moving rather than for progress."),
     ("It likely fixes the map too:", "the walk-back drift on dc22 and g50t is explained by the unmasked bar."),
     ("The third-place competition entry does this:", "it masks “probable status bars” before "
                                                     "fingerprinting screens, which its authors say "
                                                     "“substantially reduces the number of recognized "
                                                     "states” (Rudakov et al., 2025)."),
     ("Research on curiosity:", "the “noisy TV” problem, where a curious agent is drawn to something "
                                 "that changes on its own (Burda et al., 2019).")],
    ["Fixes a measured flaw on about half the games.",
     "Improves both the question and the map with one change.",
     "Cheap: about 40 microseconds per move."],
    ["Without the edge rule it wrongly masked real game pieces on 3 games, so the edge rule matters; a game "
     "whose real pieces sit at the very edge could still lose information.",
     "Our scoring tools still use the old rule, so Goose and our scores would count screens slightly "
     "differently. Changing the scoring tools would move every published number and is a team decision."],
    "The new question with bars masked, against the new question alone; and the map with bars masked, "
    "against the map alone.")

candidate(
    2, "Walk back only when it pays", "map_gated",
    "The map walks back to where Goose last found something new after every game over. The upgrade makes that "
    "a choice Goose learns per game: it tries both “walk back” and “start fresh”, scores each "
    "attempt by how many new screens it found per move, and leans towards whichever has been paying off "
    "recently. Walks to an untried spot when Goose is stuck are kept.",
    [("Our data:", "the map’s walk back is what took half the moves on the games it hurt, while it drove "
                   "the gains on tu93. The right answer differs by game."),
     ("Agent57 (DeepMind, 2020):", "the first agent to beat the human benchmark on all 57 Atari games used "
                                   "exactly this kind of per-game chooser, a sliding-window “upper "
                                   "confidence” bandit, to decide how much to explore on each game."),
     ("Early sign:", "in the short smoke test it already chose not to walk back on ft09 and vc33 and walked "
                     "back about half the time on tu93.")],
    ["Targets the exact reason the map failed.", "Keeps the map’s gains where they exist.",
     "Game overs are frequent, so it gets hundreds of samples per run to learn from."],
    ["More moving parts than a fixed rule; noisy early in a run.",
     "Scores attempts by new screens, which is not the same as progress towards a level."],
    "The map with the chooser, against the map alone and the new question alone.")

candidate(
    3, "Stop clicking what already did nothing", "deadclick",
    "If clicking a particular cell of a particular colour has done nothing four times in this level, Goose "
    "stops choosing it for the rest of the level and picks something else. The colour is part of the rule, so "
    "a button that changes colour when it becomes active can be clicked again.",
    [("Our data:", "on su15, vc33 and tn36 about half of all clicks repeat a click already known to do "
                   "nothing."),
     ("A strong 2026 open-source ARC-AGI-3 agent (BDR-Pro):", "uses this rule at the same threshold of "
                                                              "four. The version that added its contextual "
                                                              "click rules, this one among them, raised its "
                                                              "mean score by 68%, and its authors say this "
                                                              "rule \u201ccut wasted clicks enough to let "
                                                              "deep-level runs finish\u201d.")],
    ["Simple and cheap.", "Big measured waste on the click games Goose is weakest at."],
    ["Goose’s network already learns that these clicks score zero, so part of the gain may already "
     "be there.",
     "Our “don’t repeat yourself” rule hurt last time; this one is narrower (only proven "
     "no-ops, by appearance), but the risk is related."],
    "The new question with dead-click blocking, against the new question alone.")

candidate(
    4, "Treat objects, not pixels, as the things to try", "map_objects",
    "On click games there are 4,096 places to click, so the map’s idea of “something untried” "
    "was crude (a screen counted as untried until clicked 20 times). The upgrade splits each screen into "
    "objects, one per patch of a single colour, and counts a screen as untried while it has an object never "
    "clicked. On arrival it clicks the most eye-catching untried object: small, unusual colours first.",
    [("Third-place entry (Rudakov et al., 2025):", "exactly this: screens split into single-colour objects "
                                                   "in five priority tiers by size, shape and colour. It "
                                                   "solved 5 levels of vc33 in only 4,000 moves."),
     ("Our data:", "the map was weakest on click games, where it chose half of all moves.")],
    ["Makes the map sensible on click games.", "Directly supported by the best graph-based entry."],
    ["The largest code change on the list; about half a millisecond per move on busy screens.",
     "“Object” is a guess; some games have buttons made of several colours, or targets that are "
     "not separate objects."],
    "The map with object-level untried, against the map alone.")

candidate(
    5, "Use the map only when stuck", "map_stuck",
    "Keep the walk to an untried spot when Goose goes 200 moves without anything new, and turn off the walk "
    "back after a game over. This needs no new code; it is an existing switch.",
    [("Our data:", "the walk back is the suspected cause of the damage on solved games."),
     ("It is the cleanest test of that suspicion:", "if this arm keeps most of the gains and loses the damage, "
                                                     "the walk back was the problem.")],
    ["Zero new code.", "Easy to explain."],
    ["Probably gives up tu93’s gains, which relied on the walk back as well."],
    "The map without the walk back, against the map alone.")

candidate(
    6, "Give half credit for progress within an attempt", "attempt",
    "Today a screen counts as new only the first time in a level. After many attempts, the early screens of "
    "a level are all familiar, so moving forward at the start of an attempt earns nothing. The upgrade gives "
    "half credit for reaching a screen that is new in this attempt, even if it was seen in an earlier one.",
    [("Never Give Up (DeepMind, 2020):", "combined a per-episode memory with a lifetime memory and did much "
                                          "better on hard exploration games."),
     ("Henaff et al. (2023):", "per-episode and lifetime bonuses each win in different situations, and "
                               "combining them is more robust."),
     ("Our data:", "levels restart every 20 to 200 moves, so “per attempt” is a natural unit.")],
    ["Rewards making progress in every attempt.", "Cheap."],
    ["It only makes sense with the bars masked (otherwise every screen in an attempt looks new), so it is "
     "tested together with idea 1.",
     "It could reward re-walking the same path instead of exploring."],
    "Bars plus half credit, against the new question alone (and implicitly against bars alone).")

candidate(
    7, "A softer reward: more for rarer screens", "graded",
    "Instead of all or nothing, reward a move by how rarely its screen has been seen this level: full reward "
    "the first time, 0.71 the second, 0.58 the third, and so on.",
    [("A classic result:", "rewards that shrink with the square root of the visit count are the standard "
                           "“count-based” exploration bonus (Strehl and Littman, 2008; Tang et al., "
                           "2017, who also used screen fingerprints)."),
     ("Our crash last month:", "when every label is zero Goose’s predictions collapse; a graded reward "
                              "is never zero.")],
    ["One-line idea.", "Softens the late-level collapse."],
    ["Our data suggests a small effect: on the stuck games most moves go to screens seen five or more times, "
     "and the reward changes on only 5 to 20% of moves."],
    "The graded reward, against the new question alone.")

candidate(
    8, "Walk back to less-visited places", "map_diverse",
    "After a game over the map walks back to the most recent discovery. The upgrade instead picks among the "
    "reachable untried places at random, favouring ones visited less often.",
    [("Go-Explore (Nature, 2021):", "chooses where to return with weights of one over the square root of "
                                    "the visit count, and solved famously hard Atari games.")],
    ["Spreads exploration more evenly.", "Small change."],
    ["The most recent discovery is often the deepest, so this may trade depth for breadth, which hurts in "
     "games where the goal is far along."],
    "The map with this target choice, against the map alone.")

candidate(
    9, "Keep what Goose learned when it reaches a new level", "persist",
    "Today Goose’s network is reset at every new level. This keeps it. It is an existing switch.",
    [("Levels in a game share mechanics,", "so what was learned should carry over. The BDR-Pro agent carries "
                                          "information between levels and matched human pace on ar25 "
                                          "level 2.")],
    ["Zero new code."],
    ["With the old question it made Goose repeat itself 18 times more (semester 1), so the prior is "
     "negative; it is included because the new question might change that."],
    "The new question without resets, against the new question alone.")

heading("Considered and left out")
bullet("Changing learning rate, batch size or reset rules: our team rules forbid it without approval, "
       "because it would muddy every comparison.", "Tuning Goose’s settings.")
bullet("Curiosity networks such as ICM or RND add a second network and are known to be drawn to noise; "
       "the bar fix attacks the same problem more directly.", "Prediction-based curiosity.")
bullet("A pure map-and-search agent like the third-place entry would replace Goose rather than improve "
       "it; ideas 1, 4 and 5 borrow its best parts instead.", "Replacing Goose with a graph search.")
bullet("Kept for later as planned (Option 2).", "The language-model advisor.")

# ---------------------------------------------------------------------------------------------
heading("The screening sweep")
para("One sweep tries all nine ideas briefly, alongside fresh runs of the current best Goose and of the map "
     "for comparison, and ranks them. It is ready but has not been started.", after=3)
table([
    ("", ""),
    ("Arms", "12: the new-question Goose, the map, and the nine ideas"),
    ("Games", "8, two from each situation: map helped (tu93, su15), map hurt (ar25, tr87), bar drift "
              "(dc22, g50t), dead clicks (vc33), and ft09"),
    ("Length", "2 random starts × 50,000 moves per game (a proper test uses 3 × 100,000)"),
    ("Total", "192 runs, about 17 hours, four at a time"),
    ("Ranking", "levels gained over the current best Goose; then more wins than losses; then how early the "
                "levels came"),
    ("“Promising”", "more levels, more wins than losses, and no game worse on both starts"),
    ("Next step", "the best one to three promising ideas get a proper 8-game test with the rule written "
                  "first"),
], (Inches(1.3), Inches(5.4)), header=False)
para("Running four at once helps only a little: the graphics card is shared, so four runs together manage "
     "about 1.15 times the speed of one. That is why the screen uses 8 games rather than all 25.", after=3)
para("Each idea lives in its own file behind its own switch, and every line it adds to Goose is tagged, so "
     "deleting the tagged lines restores Goose exactly as it is now. We checked this byte for byte. The "
     "screen cannot change the baseline, the new question or the map.",
     "Kept separate and easy to remove.")
para("", "How to run it.", after=1)
for cmd, what in [("make upgrade-screen", "start (runs in the background; keep the computer awake)"),
                  ("make upgrade-screen-status", "progress"),
                  ("make upgrade-screen-pause", "stop after the runs in progress finish, losing nothing"),
                  ("make upgrade-screen RESUME=<manifest>", "continue where it stopped")]:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(1)
    rr = p.add_run(cmd)
    rr.font.name = "Consolas"
    rr.font.size = Pt(9.5)
    p.add_run("  " + what)
para("When it finishes, the leaderboard is written to results/screen/<date>/leaderboard.md.", after=3)

heading("Sources", size=11, before=8)
for s in [
    "ARC Prize, “ARC-AGI-3 Preview: 30-Day Learnings,” 2025. arcprize.org/blog/arc-agi-3-preview-30-day-learnings",
    "Rudakov, Shock and Cowley, “Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks,” "
    "AAAI 2026 workshop. arxiv.org/abs/2512.24156",
    "BDR-Pro, ARC-AGI-3 agent for ARC Prize 2026 (open source). github.com/BDR-Pro/arc-prize-2026-arc-agi-3",
    "Badia et al., “Agent57: Outperforming the Atari Human Benchmark,” ICML 2020. arxiv.org/abs/2003.13350",
    "Badia et al., “Never Give Up: Learning Directed Exploration Strategies,” ICLR 2020. arxiv.org/abs/2002.06038",
    "Henaff, Jiang and Raileanu, “A Study of Global and Episodic Bonuses for Exploration in Contextual "
    "MDPs,” 2023. arxiv.org/abs/2306.03236",
    "Ecoffet et al., “First return, then explore,” Nature 590, 2021; and “Go-Explore,” "
    "arxiv.org/abs/1901.10995",
    "Strehl and Littman, “An analysis of model-based Interval Estimation for Markov Decision Processes,” "
    "J. Computer and System Sciences 74, 2008",
    "Tang et al., “#Exploration: A Study of Count-Based Exploration for Deep RL,” NeurIPS 2017. "
    "arxiv.org/abs/1611.04717",
    "Burda et al., “Large-Scale Study of Curiosity-Driven Learning,” ICLR 2019. arxiv.org/abs/1808.04355",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(1)
    p.add_run(s).font.size = Pt(9)

doc.save(OUT)
print("wrote", OUT)
