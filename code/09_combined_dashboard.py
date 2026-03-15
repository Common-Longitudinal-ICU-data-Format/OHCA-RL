# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "pyarrow",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium", app_title="OHCA-RL Combined Dashboard")


# ── Cell 0: Imports & Config ──────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    import json
    import base64
    import pandas as pd
    import numpy as np
    from pathlib import Path

    from utils import setup_logging
    logger = setup_logging("09_combined_dashboard")

    project_root = Path(__file__).parent.parent.resolve()

    with open(project_root / "config" / "config.json", "r") as _f:
        config = json.load(_f)

    site_name = config["site_name"]
    intermediate_dir = project_root / "output" / "intermediate"
    training_dir = intermediate_dir / "training"
    final_dir = project_root / "output" / "final"
    fig_dir = final_dir / "figures"
    ext_val_dir = final_dir / "external_validation"

    mo.md(f"# OHCA-RL Combined Dashboard Generator\n**Site:** {site_name}")
    return (
        base64,
        config,
        ext_val_dir,
        fig_dir,
        final_dir,
        intermediate_dir,
        json,
        logger,
        mo,
        np,
        pd,
        project_root,
        site_name,
        training_dir,
    )


# ── Cell 1: Load All Outputs ─────────────────────────────────────────
@app.cell
def _(base64, ext_val_dir, fig_dir, final_dir, logger, mo, np, pd, training_dir):
    def fig_to_data_uri(path):
        """Read an image file and return a base64 data URI."""
        if not path.exists():
            return None
        _suffix = path.suffix.lower()
        _mime = "image/png" if _suffix == ".png" else "image/jpeg" if _suffix in (".jpg", ".jpeg") else "image/svg+xml" if _suffix == ".svg" else "application/pdf"
        with open(path, "rb") as _f:
            _data = base64.b64encode(_f.read()).decode("utf-8")
        return f"data:{_mime};base64,{_data}"

    def read_html_table(path):
        """Read an HTML file and extract just the content body."""
        if not path.exists():
            return "<p><em>Not available</em></p>"
        with open(path, "r", encoding="utf-8") as _f:
            _html = _f.read()
        # Extract content between <div class="container"> and closing </div>
        if '<div class="container">' in _html:
            _start = _html.index('<div class="container">')
            _end = _html.rindex("</div>")
            return _html[_start:_end + 6]
        return _html

    def read_csv_as_html_table(path, max_rows=50):
        """Read a CSV and render as styled HTML table."""
        if not path.exists():
            return "<p><em>Not available</em></p>"
        _df = pd.read_csv(path)
        if len(_df) > max_rows:
            _df = _df.head(max_rows)
        return _df.to_html(index=False, classes="data-table", border=0)

    # ── Collect all figures ──
    _fig_names = [
        "fig1_missingness_heatmap",
        "fig2_vasopressor_nee",
        "fig3_treatment_timelines",
        "fig4_sofa_trajectory",
        "fig5_vital_signs",
        "fig6_action_distribution",
        "fig7_labs",
        "fig8_training_curves",
        "fig9_action_distribution",
        "fig10_patient_timelines",
        "fig11_agreement_outcome",
        "fig12_concordance_or",
        "fig13_consort_updated",
        "fig14_action_confusion_matrix",
    ]

    figures = {}
    for _name in _fig_names:
        _png = fig_dir / f"{_name}.png"
        if _png.exists():
            figures[_name] = fig_to_data_uri(_png)
            logger.info("Loaded figure: %s", _name)
        else:
            figures[_name] = None
            logger.warning("Missing figure: %s", _name)

    # ── Load HTML tables ──
    table1_full_html = read_html_table(final_dir / "table1_ohca.html")
    table1_vaso_html = read_html_table(final_dir / "table1_ohca_vaso.html")

    # ── Load training data ──
    history_html = read_csv_as_html_table(training_dir / "training_history.csv")
    action_summary_html = read_csv_as_html_table(training_dir / "test_action_summary.csv")
    coef_html = read_csv_as_html_table(training_dir / "coef_summary.csv")
    bin_html = read_csv_as_html_table(training_dir / "bin_summary.csv")

    # ── Load STROBE counts ──
    strobe_html = read_csv_as_html_table(final_dir / "strobe_counts.csv")

    # ── Load training results summary ──
    results_summary_html = read_csv_as_html_table(final_dir / "training_results_summary.csv")

    # ── External validation ──
    has_ext_val = ext_val_dir.exists() and (ext_val_dir / "evaluation_metadata.json").exists()
    if has_ext_val:
        with open(ext_val_dir / "evaluation_metadata.json") as _f:
            import json as _json
            ext_meta = _json.load(_f)
        ext_action_html = read_csv_as_html_table(ext_val_dir / "action_summary.csv")
        ext_coef_html = read_csv_as_html_table(ext_val_dir / "coef_summary.csv")
        ext_bin_html = read_csv_as_html_table(ext_val_dir / "bin_summary.csv")
    else:
        ext_meta = {}
        ext_action_html = ""
        ext_coef_html = ""
        ext_bin_html = ""

    _n_loaded = sum(1 for v in figures.values() if v is not None)
    mo.md(f"Loaded **{_n_loaded}/{len(_fig_names)}** figures, tables, and training data.")
    return (
        action_summary_html,
        bin_html,
        coef_html,
        ext_action_html,
        ext_bin_html,
        ext_coef_html,
        ext_meta,
        fig_to_data_uri,
        figures,
        has_ext_val,
        history_html,
        results_summary_html,
        strobe_html,
        table1_full_html,
        table1_vaso_html,
    )


# ── Cell 2: Build HTML Sections ───────────────────────────────────────
@app.cell
def _(action_summary_html, bin_html, coef_html, ext_action_html, ext_bin_html, ext_coef_html, ext_meta, figures, has_ext_val, history_html, logger, results_summary_html, site_name, strobe_html, table1_full_html, table1_vaso_html):
    def _img_tag(fig_name, alt="", width="100%"):
        _uri = figures.get(fig_name)
        if _uri:
            return f'<img src="{_uri}" alt="{alt}" style="max-width:{width}; height:auto; margin: 10px 0;">'
        return f'<p class="missing"><em>Figure not available: {fig_name}</em></p>'

    # ── Section 1: Overview ──
    overview_html = f"""
    <div class="section">
        <h2>Study Overview</h2>
        <p>Out-of-Hospital Cardiac Arrest (OHCA) Reinforcement Learning study. A Double Deep Q-Network (DDQN)
        was trained to recommend vasopressor management actions for OHCA patients admitted to the ICU.</p>

        <h3>CONSORT Flow Diagram</h3>
        {_img_tag("fig13_consort_updated", "CONSORT Flow Diagram", "80%")}

        <h3>Key Results</h3>
        {results_summary_html}

        <h3>STROBE Counts</h3>
        {strobe_html}
    </div>
    """

    # ── Section 2: Table One ──
    table1_html = f"""
    <div class="section">
        <h2>Baseline Characteristics</h2>

        <h3>Full OHCA Cohort</h3>
        {table1_full_html}

        <hr style="margin: 40px 0;">

        <h3>Vasopressor Cohort (Training Population)</h3>
        {table1_vaso_html}
    </div>
    """

    # ── Section 3: Pre-Training Figures ──
    pretraining_html = f"""
    <div class="section">
        <h2>Pre-Training Figures</h2>

        <h3>Figure 1: Missingness Heatmap</h3>
        {_img_tag("fig1_missingness_heatmap", "Missingness Heatmap")}

        <h3>Figure 2: Vasopressor/NEE Temporal</h3>
        {_img_tag("fig2_vasopressor_nee", "Vasopressor/NEE")}

        <h3>Figure 3: Treatment Timelines</h3>
        {_img_tag("fig3_treatment_timelines", "Treatment Timelines")}

        <h3>Figure 4: SOFA Trajectory</h3>
        {_img_tag("fig4_sofa_trajectory", "SOFA Trajectory")}

        <h3>Figure 5: Vital Signs</h3>
        {_img_tag("fig5_vital_signs", "Vital Signs")}

        <h3>Figure 6: Clinician Action Distribution Over Time</h3>
        {_img_tag("fig6_action_distribution", "Action Distribution")}

        <h3>Figure 7: Lab Trajectories</h3>
        {_img_tag("fig7_labs", "Lab Trajectories")}
    </div>
    """

    # ── Section 4: Training Results ──
    training_html = f"""
    <div class="section">
        <h2>Training Results</h2>

        <h3>Figure 8: Training Loss Curves</h3>
        {_img_tag("fig8_training_curves", "Training Curves")}

        <h3>Training History</h3>
        {history_html}

        <h3>Figure 9: Action Distribution (RL Agent vs Clinician)</h3>
        {_img_tag("fig9_action_distribution", "Action Distribution Comparison")}

        <h3>Action Summary</h3>
        {action_summary_html}
    </div>
    """

    # ── Section 5: Evaluation ──
    evaluation_html = f"""
    <div class="section">
        <h2>Model Evaluation</h2>

        <h3>Figure 10: Patient Timeline Heatmaps</h3>
        {_img_tag("fig10_patient_timelines", "Patient Timelines")}

        <h3>Figure 14: Action Confusion Matrix</h3>
        {_img_tag("fig14_action_confusion_matrix", "Action Confusion Matrix")}

        <h3>Figure 11: Agreement-Outcome Relationship</h3>
        {_img_tag("fig11_agreement_outcome", "Agreement-Outcome")}

        <h3>Agreement Bins</h3>
        {bin_html}

        <h3>Figure 12: Concordance OR Forest Plot</h3>
        {_img_tag("fig12_concordance_or", "Concordance OR")}

        <h3>Ordinal Logistic Regression</h3>
        {coef_html}
    </div>
    """

    # ── Section 6: External Validation ──
    if has_ext_val:
        _ext_agreement = ext_meta.get("overall_agreement", "N/A")
        _ext_or = ext_meta.get("ordinal_or", "N/A")
        _ext_n = ext_meta.get("n_patients", "N/A")
        _ext_site = ext_meta.get("local_site", "Unknown")
        _ext_train_site = ext_meta.get("training_site", "Unknown")

        external_html = f"""
        <div class="section">
            <h2>External Validation</h2>
            <p>Training site: <strong>{_ext_train_site}</strong> | Validation site: <strong>{_ext_site}</strong></p>

            <table class="data-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>N patients</td><td>{_ext_n}</td></tr>
                <tr><td>Overall Agreement</td><td>{_ext_agreement:.1%}</td></tr>
                <tr><td>Concordance OR</td><td>{_ext_or:.1f}</td></tr>
            </table>

            <h3>Action Summary</h3>
            {ext_action_html}

            <h3>Agreement Bins</h3>
            {ext_bin_html}

            <h3>Ordinal Logistic Regression</h3>
            {ext_coef_html}
        </div>
        """
    else:
        external_html = """
        <div class="section">
            <h2>External Validation</h2>
            <p><em>No external validation results available. Run <code>07_external_validation.py</code>
            with shared model artifacts to generate these results.</em></p>
        </div>
        """

    logger.info("Built all HTML sections")
    return (
        evaluation_html,
        external_html,
        overview_html,
        pretraining_html,
        table1_html,
        training_html,
    )


# ── Cell 3: Assemble Dashboard HTML ───────────────────────────────────
@app.cell
def _(evaluation_html, external_html, logger, mo, overview_html, pd, pretraining_html, site_name, table1_html, training_html):
    _tabs = [
        ("overview", "Overview", overview_html),
        ("table1", "Table One", table1_html),
        ("pretraining", "Pre-Training", pretraining_html),
        ("training", "Training", training_html),
        ("evaluation", "Evaluation", evaluation_html),
        ("external", "External Validation", external_html),
    ]

    _tab_buttons = "\n".join(
        f'        <button class="tab-btn{" active" if i == 0 else ""}" '
        f'onclick="switchTab(\'{tid}\')" id="btn-{tid}">{label}</button>'
        for i, (tid, label, _) in enumerate(_tabs)
    )

    _tab_contents = "\n".join(
        f'    <div class="tab-content{" active" if i == 0 else ""}" id="tab-{tid}">\n{content}\n    </div>'
        for i, (tid, _, content) in enumerate(_tabs)
    )

    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OHCA-RL Dashboard — {site_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #1565C0, #0D47A1);
            color: white;
            padding: 25px 40px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .header h1 {{ font-size: 24px; font-weight: 600; }}
        .header p {{ font-size: 13px; opacity: 0.85; margin-top: 5px; }}
        .tab-bar {{
            background: white;
            padding: 0 40px;
            display: flex;
            gap: 0;
            border-bottom: 2px solid #e0e0e0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            overflow-x: auto;
        }}
        .tab-btn {{
            padding: 14px 24px;
            border: none;
            background: none;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .tab-btn:hover {{ color: #1565C0; background: #f5f8ff; }}
        .tab-btn.active {{
            color: #1565C0;
            border-bottom-color: #1565C0;
            font-weight: 600;
        }}
        .content {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px 40px;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }}
        .section h2 {{
            color: #1565C0;
            font-size: 20px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e3f2fd;
        }}
        .section h3 {{
            color: #333;
            font-size: 15px;
            margin: 25px 0 10px 0;
        }}
        .section img {{
            display: block;
            margin: 15px auto;
            border: 1px solid #eee;
            border-radius: 4px;
        }}
        table, .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0 20px 0;
            font-size: 12px;
        }}
        th {{
            background: #1565C0;
            color: white;
            padding: 10px 12px;
            text-align: left;
            font-weight: 500;
            border: 1px solid #1256A0;
        }}
        td {{
            padding: 8px 12px;
            border: 1px solid #e0e0e0;
            vertical-align: top;
        }}
        tr:nth-child(even) {{ background: #fafafa; }}
        tr:hover {{ background: #f0f7ff; }}
        .missing {{ color: #999; font-style: italic; }}
        hr {{ border: none; border-top: 1px solid #eee; }}
        .footer {{
            text-align: center;
            padding: 20px;
            font-size: 11px;
            color: #999;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OHCA-RL Results Dashboard</h1>
        <p>{site_name} | Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="tab-bar">
{_tab_buttons}
    </div>

    <div class="content">
{_tab_contents}
    </div>

    <div class="footer">
        <p>OHCA-RL: Out-of-Hospital Cardiac Arrest Reinforcement Learning</p>
    </div>

    <script>
    function switchTab(tabId) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-' + tabId).classList.add('active');
        document.getElementById('btn-' + tabId).classList.add('active');
    }}
    </script>
</body>
</html>"""

    logger.info("Assembled dashboard HTML: %d characters", len(dashboard_html))
    mo.md(f"Dashboard HTML assembled: **{len(dashboard_html):,}** characters")
    return (dashboard_html,)


# ── Cell 4: Save Dashboard ────────────────────────────────────────────
@app.cell
def _(dashboard_html, final_dir, logger, mo):
    _dashboard_path = final_dir / "ohca_rl_dashboard.html"
    with open(_dashboard_path, "w", encoding="utf-8") as _f:
        _f.write(dashboard_html)

    _size_mb = _dashboard_path.stat().st_size / 1024 / 1024

    logger.info("Saved dashboard: %s (%.1f MB)", _dashboard_path, _size_mb)

    mo.md(f"""
    ## Dashboard Saved

    | | |
    |---|---|
    | **Path** | `{_dashboard_path}` |
    | **Size** | {_size_mb:.1f} MB |

    Open in a browser to view the interactive tabbed dashboard.
    """)
    return


if __name__ == "__main__":
    app.run()
