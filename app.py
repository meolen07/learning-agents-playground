"""Streamlit dashboard for visualizing Learning Agents Playground results.

This app is visualization-only — training is performed via CLI scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation.metrics import moving_average, summarize_rewards, success_rate

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def list_run_directories(results_dir: Path) -> list[Path]:
    """Return subdirectories that contain a training log CSV."""
    if not results_dir.exists():
        return []
    runs = []
    for path in sorted(results_dir.iterdir()):
        if path.is_dir() and (path / "training_log.csv").exists():
            runs.append(path)
    return runs


def load_run_data(run_dir: Path) -> pd.DataFrame:
    """Load training log CSV for a run."""
    return pd.read_csv(run_dir / "training_log.csv")


def main() -> None:
    st.set_page_config(
        page_title="Learning Agents Playground",
        page_icon="🎮",
        layout="wide",
    )

    st.title("Learning Agents Playground")
    st.caption("Visualize RL training logs — author: Huynh Mai Linh Nguyen")

    st.sidebar.header("Run Selection")
    results_dir = st.sidebar.text_input("Results directory", str(DEFAULT_RESULTS_DIR))
    results_path = Path(results_dir)

    runs = list_run_directories(results_path)
    if not runs:
        st.info(
            "No training runs found. Train an agent first:\n\n"
            "```bash\n"
            "python scripts/train_q_learning.py --verbose\n"
            "python scripts/train_dqn.py --verbose\n"
            "```"
        )
        return

    run_names = [run.name for run in runs]
    selected_name = st.sidebar.selectbox("Training run", run_names)
    run_dir = results_path / selected_name
    df = load_run_data(run_dir)

    st.subheader(f"Run: `{selected_name}`")
    col1, col2, col3, col4 = st.columns(4)
    summary = summarize_rewards(df["reward"].tolist())
    col1.metric("Episodes", int(summary["count"]))
    col2.metric("Mean reward", f"{summary['mean']:.2f}")
    col3.metric("Max reward", f"{summary['max']:.2f}")
    if "success" in df.columns:
        col4.metric("Success rate", f"{success_rate(df['success'].astype(bool)):.1%}")

    window = st.sidebar.slider("Moving-average window", min_value=5, max_value=200, value=50)

    tab_rewards, tab_success, tab_table, tab_assets = st.tabs(
        ["Rewards", "Success Rate", "Raw Data", "Saved Assets"]
    )

    with tab_rewards:
        st.line_chart(
            pd.DataFrame(
                {
                    "reward": df["reward"],
                    "moving_avg": moving_average(df["reward"].tolist(), window),
                },
                index=df["episode"],
            )
        )
        reward_plot = run_dir / "rewards.png"
        if reward_plot.exists():
            st.image(str(reward_plot), caption="Saved reward plot")

    with tab_success:
        if "success" in df.columns:
            st.line_chart(
                pd.DataFrame(
                    {
                        "success_rate": moving_average(
                            df["success"].astype(float).tolist(), window
                        )
                    },
                    index=df["episode"],
                )
            )
            success_plot = run_dir / "success_rate.png"
            if success_plot.exists():
                st.image(str(success_plot), caption="Saved success-rate plot")
        else:
            st.write("No success column in this run's CSV.")

    with tab_table:
        st.dataframe(df, use_container_width=True)

    with tab_assets:
        st.write("Files in run directory:")
        for file_path in sorted(run_dir.iterdir()):
            if file_path.is_file():
                st.write(f"- `{file_path.name}`")
                if file_path.suffix == ".png":
                    st.image(str(file_path), caption=file_path.name)


if __name__ == "__main__":
    main()
