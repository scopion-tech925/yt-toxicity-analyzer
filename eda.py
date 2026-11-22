"""EDA: toxicity rate by channel / topic, and toxicity vs. view-count relationship."""
import os, sqlite3
import matplotlib
matplotlib.use("Agg")                      # headless-safe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import config


def load_frames():
    con = sqlite3.connect(config.DB_PATH)
    comments = pd.read_sql("SELECT * FROM comments", con)
    videos   = pd.read_sql("SELECT * FROM videos",   con)
    con.close()
    df = comments.merge(videos[["video_id", "channel", "view_count", "video_title"]],
                        on="video_id", how="left")
    return df


def by_channel(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby("channel")
           .agg(comments=("is_toxic", "size"),
                toxic=("is_toxic", "sum"),
                mean_toxicity=("toxicity_score", "mean"),
                median_views=("view_count", "median"))
           .assign(toxicity_rate=lambda x: x["toxic"] / x["comments"])
           .sort_values("toxicity_rate", ascending=False))
    return g.round(4)


def by_topic(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby("topic")
           .agg(comments=("is_toxic", "size"), toxic=("is_toxic", "sum"))
           .assign(toxicity_rate=lambda x: x["toxic"] / x["comments"])
           .query("comments >= 5")                      # filter tiny samples
           .sort_values("toxicity_rate", ascending=False))
    return g.round(4)


def toxicity_vs_views(df: pd.DataFrame):
    """Two lenses: per-video rate vs. video views; per-comment score vs. views."""
    per_video = (df.groupby(["video_id", "channel"])
                   .agg(view_count=("view_count", "first"),
                        toxicity_rate=("is_toxic", "mean"),
                        n_comments=("is_toxic", "size"))
                   .reset_index())
    # rate needs enough comments and a known view count; plot uses the SAME subset
    pv = per_video[(per_video["n_comments"] >= 10) & per_video["view_count"].notna()]
    res = {}
    if len(pv) >= 3:
        rho, p = stats.spearmanr(np.log1p(pv["view_count"]), pv["toxicity_rate"])
        res["video_level"] = {"spearman_rho": round(rho, 3), "p_value": round(p, 4),
                              "n_videos": len(pv)}
    # NOTE: view_count is constant within a video, so comments are not independent;
    # this p-value is optimistic. Treat the video-level result as the primary one.
    cl = df.dropna(subset=["view_count"])
    rho2, p2 = stats.spearmanr(np.log1p(cl["view_count"]), cl["toxicity_score"])
    res["comment_level"] = {"spearman_rho": round(rho2, 3), "p_value": round(p2, 4),
                            "n_comments": len(cl)}
    return res, pv


def make_plots(df, chan, pv, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    chan["toxicity_rate"].plot.bar(ax=axes[0], color="#e4572e")
    axes[0].set_title("Toxicity rate by channel"); axes[0].set_ylabel("rate")
    axes[0].tick_params(axis="x", rotation=20)

    ax = axes[1]
    for ch, sub in pv.groupby("channel"):
        ax.scatter(np.log1p(sub["view_count"]), sub["toxicity_rate"],
                   label=ch, alpha=0.7)
    lx = np.log1p(pv["view_count"])
    if len(pv) >= 3 and lx.nunique() > 1:               # polyfit needs spread in x
        z = np.polyfit(lx, pv["toxicity_rate"], 1)
        xs = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.6)
    ax.set_title("Video views (log) vs. comment toxicity rate")
    ax.set_xlabel("log(1 + views)"); ax.set_ylabel("toxicity rate"); ax.legend(fontsize=8, loc="lower right")

    g = df.groupby("topic")["toxicity_score"].agg(["mean", "size"])
    top = g[g["size"] >= 5]["mean"].nlargest(10).sort_values()   # same n>=5 rule as by_topic
    top.plot.barh(ax=axes[2], color="#2e86ab")
    axes[2].set_title("Mean toxicity score by topic (top 10)")

    fig.tight_layout()
    path = os.path.join(outdir, "toxicity_eda.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def run():
    df = load_frames()
    if df.empty:
        raise SystemExit("No comments in the database - run extract/load first.")
    chan = by_channel(df);            print("\n=== TOXICITY BY CHANNEL ===\n", chan)
    top  = by_topic(df);              print("\n=== TOXICITY BY TOPIC (n>=5) ===\n", top.head(10))
    res, pv = toxicity_vs_views(df)
    print("\n=== TOXICITY vs VIEWS (Spearman, views log-transformed) ===")
    for lvl, r in res.items():
        print(f"  {lvl}: rho={r['spearman_rho']}  p={r['p_value']}  n={list(r.values())[-1]}")

    os.makedirs(config.REPORT_DIR, exist_ok=True)
    chan.to_csv(os.path.join(config.REPORT_DIR, "by_channel.csv"))
    top.to_csv(os.path.join(config.REPORT_DIR, "by_topic.csv"))
    pd.DataFrame(res).T.to_csv(os.path.join(config.REPORT_DIR, "toxicity_vs_views.csv"))
    path = make_plots(df, chan, pv, config.REPORT_DIR)
    print(f"\nEDA reports -> {config.REPORT_DIR} (chart: {path})")


if __name__ == "__main__":
    run()
