"""NVTX self-time summary from nsys sqlite exports.

For each `trace_<tag>.sqlite` it finds:
- builds per-thread NVTX containment stack (push/pop is strictly nested)
- self-time = own duration - sum of direct children duration
- prints a text table + saves one bar plot (% of wall per range).
"""
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

RUNS = {"short": "../demo/sample.wav", "long": "../demo/B047.wav"}
TOP_N = 15


def fmt(s):
    if s < 60: return f"{s:.2f}s"
    m, s = divmod(s, 60)
    return f"{int(m)}m{int(s):02d}s"


def self_times(sqlite_path):
    """Return (wall_seconds, {range_name: self_seconds})."""
    # nsys records the cudaProfilerApi-wrapper events (`diarize`, the outer
    # `get_embeddings`/WeSpeakerResNet34 entries) on the *session* clock while
    # nn.Module-hook NVTX events use a *process-relative* clock. Both span the
    # same window; we keep only process-relative (start < 1e12 ns = 1000s) and
    # derive wall from min(start)/max(end) of those.
    conn = sqlite3.connect(sqlite_path)
    rows = conn.execute(
        "SELECT text, start, end, globalTid FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND text IS NOT NULL AND start < 1e12"
    ).fetchall()
    conn.close()
    if not rows: return 0.0, {}

    wall = (max(e for _, _, e, _ in rows) - min(s for _, s, _, _ in rows)) / 1e9

    by_thread = {}
    for text, s, e, tid in rows:
        by_thread.setdefault(tid, []).append((s, e, text))

    self_ns = {}
    for events in by_thread.values():
        events.sort(key=lambda x: (x[0], -x[1]))  # parents before children on ties
        stack = []  # [start, end, name, children_sum]
        for s, e, name in events:
            while stack and stack[-1][1] <= s:
                d = stack.pop()
                self_ns[d[2]] = self_ns.get(d[2], 0) + (d[1] - d[0] - d[3])
            if stack: stack[-1][3] += e - s
            stack.append([s, e, name, 0])
        while stack:
            d = stack.pop()
            self_ns[d[2]] = self_ns.get(d[2], 0) + (d[1] - d[0] - d[3])

    return wall, {n: v / 1e9 for n, v in self_ns.items()}


# --- Gather ---
runs = {}
for tag, audio in RUNS.items():
    path = Path(f"trace_{tag}.sqlite")
    if not path.exists():
        print(f"skip {tag}: {path} missing")
        continue
    audio_s = sf.info(audio).frames / sf.info(audio).samplerate
    wall, st = self_times(path)
    runs[tag] = dict(audio=audio_s, wall=wall, self=st)

if not runs:
    raise SystemExit("no traces; run `just nsys` first")

cols = list(runs)
top = sorted({n for r in runs.values() for n in r["self"]},
             key=lambda n: -sum(r["self"].get(n, 0) for r in runs.values()))[:TOP_N]

# --- Text table ---
W = 14
print(f"{'metric':<22}" + "".join(f"{c:>{W}}" for c in cols))
print("-" * (22 + W * len(cols)))
print(f"{'audio':<22}" + "".join(f"{fmt(runs[c]['audio']):>{W}}" for c in cols))
print(f"{'wall':<22}" + "".join(f"{fmt(runs[c]['wall']):>{W}}" for c in cols))
print(f"{'x real-time':<22}" + "".join(f"{runs[c]['audio']/runs[c]['wall']:>{W}.1f}" for c in cols))
print(f"{'self-time coverage':<22}"
      + "".join(f"{sum(runs[c]['self'].values())/runs[c]['wall']*100:>{W-1}.1f}%" for c in cols))

print(f"\n{'range (self / %wall)':<22}" + "".join(f"{c:>{W}}" for c in cols))
print("-" * (22 + W * len(cols)))
for n in top:
    line = f"{n[:21]:<22}"
    for c in cols:
        v = runs[c]["self"].get(n, 0)
        cell = f"{fmt(v)} {v/runs[c]['wall']*100:>4.1f}%" if v > 0 else "-"
        line += f"{cell:>{W}}"
    print(line)

# --- One plot: % of wall per range, grouped bars ---
display = sorted(top, key=lambda n: sum(runs[c]["self"].get(n, 0) / runs[c]["wall"] for c in cols))
y = np.arange(len(display))
h = 0.8 / len(cols)
fig, ax = plt.subplots(figsize=(10, 7))
for i, c in enumerate(cols):
    pcts = [runs[c]["self"].get(n, 0) / runs[c]["wall"] * 100 for n in display]
    bars = ax.barh(y + (i - (len(cols)-1)/2) * h, pcts, height=h * 0.9,
                   label=f"{c} ({fmt(runs[c]['audio'])})")
    for b, v in zip(bars, pcts):
        if v >= 0.05:
            ax.text(v, b.get_y() + b.get_height()/2, f" {v:.1f}%",
                    va="center", fontsize=7, clip_on=False)
ax.set_yticks(y); ax.set_yticklabels(display)
ax.set_xlabel("% of wall time"); ax.set_xlim(0, 110)
ax.set_title("Diarization self-time share")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
out = Path("baseline.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nplot: {out}")
