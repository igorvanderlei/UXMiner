from pathlib import Path
import json
import html
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATES_DIR = PROJECT_ROOT / "data" / "states"
CLICKS_DIR = PROJECT_ROOT / "data" / "clicks"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def find_states_json() -> Path:
    semantic = sorted(STATES_DIR.glob("*_semantic_states.json"))
    if semantic:
        return semantic[0]

    plain = sorted(STATES_DIR.glob("*_states.json"))
    if plain:
        return plain[0]

    raise FileNotFoundError(f"No states file found in {STATES_DIR}")


def find_clicks_json(video_name: str) -> Path | None:
    expected = CLICKS_DIR / f"{video_name}_clicks.json"

    if expected.exists():
        return expected

    matches = sorted(CLICKS_DIR.glob("*_clicks.json"))

    if matches:
        return matches[0]

    return None


def load_clicks(video_name: str) -> dict:
    clicks_json = find_clicks_json(video_name)

    if clicks_json is None:
        return {
            "available": False,
            "clicks": [],
            "summary_by_task": []
        }

    with clicks_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    data["available"] = True
    data["source_file"] = str(clicks_json)

    return data


def copy_image_to_report(image_path: Path, assets_dir: Path) -> str:
    assets_dir.mkdir(parents=True, exist_ok=True)

    dst = assets_dir / image_path.name
    shutil.copy2(image_path, dst)

    return f"assets/{dst.name}"


def esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def render_keywords(keywords: list[str]) -> str:
    if not keywords:
        return ""

    items = "".join(
        f"<span class='tag'>{esc(k)}</span>"
        for k in keywords
    )

    return f"<div class='tags'>{items}</div>"


def render_input_context(input_context: dict) -> str:
    if not input_context:
        return ""

    detected = input_context.get("detected_text", [])
    masked = input_context.get("masked_input_detected", False)

    parts = []

    if detected:
        parts.append(
            "<div><strong>Detected input:</strong> "
            + ", ".join(esc(x) for x in detected)
            + "</div>"
        )

    if masked:
        parts.append("<div><strong>Masked input detected:</strong> yes</div>")

    if not parts:
        return ""

    return "<div class='input-context'>" + "".join(parts) + "</div>"


def is_double_click(click: dict) -> bool:
    return bool(click.get("double_click_group"))


def clicks_for_task(clicks: list[dict], task_id: str) -> list[dict]:
    return [
        click for click in clicks
        if click.get("task_id") == task_id
    ]


def count_double_click_groups(task_clicks: list[dict]) -> int:
    groups = {
        click.get("double_click_group")
        for click in task_clicks
        if click.get("double_click_group")
    }

    return len(groups)


def count_visual_clicks(clicks: list[dict]) -> int:
    count = 0
    seen_double_groups = set()

    for click in clicks:
        group = click.get("double_click_group")

        if group:
            if group in seen_double_groups:
                continue

            seen_double_groups.add(group)
            count += 1
        else:
            count += 1

    return count


def clicks_for_transition(task_clicks: list[dict], transition: dict) -> list[dict]:
    start = transition["start_timestamp"]
    end = transition["end_timestamp"]

    return [
        click for click in task_clicks
        if start <= click["timestamp_seconds"] <= end
    ]


def render_click_timeline(
    transition_clicks: list[dict],
    transition: dict
) -> str:
    if not transition_clicks:
        return "<div class='click-timeline empty'>No clicks detected in this transition.</div>"

    start = transition["start_timestamp"]
    end = transition["end_timestamp"]
    duration = max(end - start, 0.001)

    markers = []
    rendered_double_groups = set()

    for click in transition_clicks:
        double = is_double_click(click)

        if double:
            group = click.get("double_click_group")

            if group in rendered_double_groups:
                continue

            rendered_double_groups.add(group)

        pos = ((click["timestamp_seconds"] - start) / duration) * 100
        pos = max(0, min(100, pos))

        klass = "click-marker double" if double else "click-marker single"

        title = (
            f"{click.get('click_id', '')} | "
            f"{click['relative_timestamp_seconds']:.2f}s relative | "
            f"x={click.get('x')} y={click.get('y')} | "
            f"{click.get('detector', '')}"
        )

        if double:
            title += f" | double_click={click.get('double_click_group')}"

        markers.append(
            f"<span class='{klass}' style='left:{pos:.2f}%' title='{esc(title)}'></span>"
        )

    return f"""
    <div class="click-timeline">
        {''.join(markers)}
    </div>
    """


def render_transition_clicks(task_clicks: list[dict], transitions: list[dict]) -> str:
    if not transitions:
        return ""

    rows = []

    for tr in transitions:
        tr_clicks = clicks_for_transition(task_clicks, tr)

        visual_clicks = count_visual_clicks(tr_clicks)
        single_events = sum(1 for c in tr_clicks if not is_double_click(c))
        double_groups = count_double_click_groups(tr_clicks)

        rows.append(f"""
        <div class="transition-click-row">
            <div class="transition-click-header">
                <code>{esc(tr['source'])} → {esc(tr['target'])}</code>
                <span>{tr['duration_seconds']:.2f}s</span>
                <span>{visual_clicks} visual clicks</span>
                <span>{single_events} single</span>
                <span>{double_groups} double</span>
            </div>
            {render_click_timeline(tr_clicks, tr)}
        </div>
        """)

    return f"""
    <div class="clicks-by-transition">
        <h3>Click timeline by state transition</h3>
        <div class="click-legend-small task-level-legend">
            <span class="legend-dot single"></span> single click
            <span class="legend-dot double"></span> double click
        </div>
        {''.join(rows)}
    </div>
    """


def build_html(data: dict, clicks_data: dict, assets_dir: Path) -> str:
    all_clicks = clicks_data.get("clicks", [])
    click_source = clicks_data.get("source_file", "not available")

    tasks_html = []

    for task in data["tasks"]:
        task_clicks = clicks_for_task(all_clicks, task["task_id"])

        total_clicks = len(task_clicks)
        double_click_groups = count_double_click_groups(task_clicks)
        simple_clicks = sum(1 for c in task_clicks if not is_double_click(c))

        states_html = []

        for idx, state in enumerate(task["states"], start=1):
            image_path = Path(state["representative_frame"])
            image_rel = copy_image_to_report(image_path, assets_dir)

            label = (
                state.get("semantic_label")
                or state.get("ocr_summary")
                or state["state_id"]
            )

            summary = (
                state.get("semantic_summary")
                or state.get("ocr_summary")
                or ""
            )

            keywords = state.get("semantic_keywords", [])
            input_context = state.get("input_context", {})
            detection = state.get("state_detection", {})

            reason = detection.get("reason", "")
            relative_time = state.get("relative_timestamp_seconds", 0)

            states_html.append(f"""
            <article class="state-card">
                <div class="thumb-wrap">
                    <img src="{esc(image_rel)}" alt="{esc(state['state_id'])}">
                </div>

                <div class="state-info">
                    <div class="state-header">
                        <span class="state-index">#{idx}</span>
                        <span class="state-id">{esc(state['state_id'])}</span>
                        <span class="state-time">{relative_time:.2f}s</span>
                    </div>

                    <h3>{esc(label)}</h3>
                    <p class="summary">{esc(summary)}</p>

                    {render_keywords(keywords)}
                    {render_input_context(input_context)}

                    <details>
                        <summary>Technical details</summary>
                        <div class="details-grid">
                            <div><strong>Frame:</strong> {esc(Path(state.get("representative_frame", "")).name)}</div>
                            <div><strong>Detection reason:</strong> {esc(reason)}</div>
                            <div><strong>OCR fingerprint:</strong> {esc(state.get("ocr_fingerprint", ""))}</div>
                        </div>
                        <pre>{esc(state.get("ocr_summary", ""))}</pre>
                    </details>
                </div>
            </article>
            """)

        transitions_html = []

        for tr in task.get("transitions", []):
            tr_clicks = clicks_for_transition(task_clicks, tr)
            visual_clicks = count_visual_clicks(tr_clicks)

            transitions_html.append(
                f"<li><code>{esc(tr['source'])} → {esc(tr['target'])}</code> "
                f"<strong>{tr['duration_seconds']:.2f}s</strong> "
                f"<span class='inline-clicks'>({visual_clicks} visual clicks)</span></li>"
            )

        tasks_html.append(f"""
        <section class="task">
            <header class="task-header">
                <h2>{esc(task['task_id'])}</h2>
                <div class="task-meta">
                    <span>Duration: {task['duration_seconds']:.2f}s</span>
                    <span>Frames: {esc(task.get('frames_count', ''))}</span>
                    <span>States: {task['states_count']}</span>
                    <span>Transitions: {task['transitions_count']}</span>
                    <span>Click events: {total_clicks}</span>
                    <span>Single click events: {simple_clicks}</span>
                    <span>Double-click groups: {double_click_groups}</span>
                </div>
            </header>

            <div class="states-list">
                {''.join(states_html)}
            </div>

            <div class="transitions">
                <h3>Transitions</h3>
                <ul>
                    {''.join(transitions_html)}
                </ul>
            </div>

            {render_transition_clicks(task_clicks, task.get("transitions", []))}
        </section>
        """)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UXMiner - Observable States Report</title>
<style>
    body {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f5f5f7;
        color: #222;
        margin: 0;
        padding: 32px;
    }}

    h1 {{
        margin: 0 0 8px 0;
        font-size: 32px;
    }}

    .subtitle {{
        color: #666;
        margin-bottom: 12px;
    }}

    .source-box {{
        background: white;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 32px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        color: #555;
        font-size: 14px;
    }}

    .task {{
        background: white;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 32px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }}

    .task-header {{
        border-bottom: 1px solid #ddd;
        margin-bottom: 24px;
        padding-bottom: 16px;
    }}

    .task-header h2 {{
        margin: 0 0 12px 0;
        font-size: 26px;
    }}

    .task-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        color: #555;
    }}

    .task-meta span {{
        background: #f0f0f2;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 14px;
    }}

    .states-list {{
        display: flex;
        flex-direction: column;
        gap: 22px;
    }}

    .state-card {{
        display: grid;
        grid-template-columns: 360px 1fr;
        gap: 22px;
        align-items: start;
        background: #fafafa;
        border: 1px solid #e3e3e6;
        border-radius: 16px;
        padding: 16px;
    }}

    .thumb-wrap {{
        background: #111;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #ccc;
    }}

    .thumb-wrap img {{
        display: block;
        width: 100%;
        height: auto;
    }}

    .state-header {{
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 8px;
        font-size: 14px;
    }}

    .state-index {{
        background: #222;
        color: white;
        border-radius: 999px;
        padding: 3px 8px;
    }}

    .state-id {{
        font-family: monospace;
        background: #e8e8ec;
        border-radius: 6px;
        padding: 3px 7px;
    }}

    .state-time {{
        color: #666;
    }}

    .state-info h3 {{
        margin: 6px 0;
        font-size: 24px;
    }}

    .summary {{
        margin: 0 0 12px 0;
        color: #444;
        font-size: 16px;
    }}

    .tags {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0;
    }}

    .tag {{
        background: #dfefff;
        color: #17456b;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 13px;
    }}

    .input-context {{
        background: #fff8df;
        border: 1px solid #eedb95;
        padding: 10px;
        border-radius: 10px;
        margin: 12px 0;
    }}

    details {{
        margin-top: 12px;
    }}

    summary {{
        cursor: pointer;
        color: #444;
    }}

    pre {{
        white-space: pre-wrap;
        background: #f0f0f0;
        border-radius: 8px;
        padding: 10px;
        font-size: 13px;
    }}

    .details-grid {{
        display: grid;
        gap: 6px;
        margin: 10px 0;
        color: #555;
        font-size: 14px;
    }}

    .transitions {{
        margin-top: 24px;
        border-top: 1px solid #ddd;
        padding-top: 16px;
    }}

    .transitions ul {{
        margin: 0;
        padding-left: 22px;
    }}

    .transitions li {{
        margin: 6px 0;
    }}

    .inline-clicks {{
        color: #777;
        font-size: 13px;
    }}

    .clicks-by-transition {{
        margin-top: 24px;
        border-top: 1px solid #ddd;
        padding-top: 16px;
    }}

    .transition-click-row {{
        background: #f8f8fa;
        border: 1px solid #e2e2e8;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
    }}

    .transition-click-header {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin-bottom: 10px;
        font-size: 14px;
        color: #555;
    }}

    .transition-click-header code {{
        background: #eee;
        padding: 3px 6px;
        border-radius: 6px;
        color: #222;
    }}

    .click-timeline {{
        position: relative;
        height: 30px;
        background: linear-gradient(to right, #e9e9ed, #f8f8fa);
        border: 1px solid #d4d4dc;
        border-radius: 999px;
        overflow: hidden;
    }}

    .click-timeline.empty {{
        height: auto;
        border-radius: 8px;
        padding: 8px 10px;
        color: #777;
        background: #f0f0f2;
        font-size: 13px;
    }}

    .click-marker {{
        position: absolute;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 11px;
        height: 11px;
        border-radius: 999px;
        border: 2px solid white;
        box-shadow: 0 1px 4px rgba(0,0,0,0.35);
    }}

    .click-marker.single {{
        background: #1f77b4;
    }}

    .click-marker.double {{
        background: #d62728;
        width: 14px;
        height: 14px;
    }}

    .click-legend-small {{
        display: flex;
        gap: 14px;
        align-items: center;
        margin-top: 6px;
        font-size: 12px;
        color: #666;
    }}

    .task-level-legend {{
        margin-bottom: 12px;
    }}

    .legend-dot {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 999px;
        margin-right: 4px;
    }}

    .legend-dot.single {{
        background: #1f77b4;
    }}

    .legend-dot.double {{
        background: #d62728;
    }}

    @media (max-width: 900px) {{
        .state-card {{
            grid-template-columns: 1fr;
        }}
    }}
</style>
</head>
<body>
    <h1>UXMiner - Observable States Report</h1>

    <div class="subtitle">
        States source: {esc(data.get("source_ocr_json", ""))}
    </div>

    <div class="source-box">
        <div><strong>Clicks file:</strong> {esc(click_source)}</div>
        <div><strong>Clicks available:</strong> {"yes" if clicks_data.get("available") else "no"}</div>
    </div>

    {''.join(tasks_html)}
</body>
</html>
"""


def main() -> None:
    states_json = find_states_json()

    with states_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    video_name = (
        states_json.name
        .replace("_semantic_states.json", "")
        .replace("_states.json", "")
    )

    clicks_data = load_clicks(video_name)

    report_dir = REPORTS_DIR / video_name
    assets_dir = report_dir / "assets"

    if report_dir.exists():
        shutil.rmtree(report_dir)

    report_dir.mkdir(parents=True, exist_ok=True)

    html_content = build_html(data, clicks_data, assets_dir)

    output_path = report_dir / "states_report.html"

    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    print("HTML report generated.")
    print(output_path)


if __name__ == "__main__":
    main()
