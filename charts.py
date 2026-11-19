import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from telemetry import reference_events


def save_timeline(data, alerts, path, *, title, highlight=None):
    channels = ["Pressure", "Volume Flow RateRMS", "Accelerometer1RMS"]
    fig, axes = plt.subplots(len(channels), 1, figsize=(9, 6), sharex=True, layout="constrained")
    labels = ["Pressure (bar)", "Flow rate (L/min)", "Vibration RMS (g)"]
    reference = reference_events(data) if "anomaly" in data else []
    change_points = data.loc[data["changepoint"], "datetime"] if "changepoint" in data else []
    for ax, channel, label in zip(axes, channels, labels, strict=True):
        ax.plot(data["datetime"], data[channel], color="#214B72", lw=0.9)
        for start, end in reference:
            ax.axvspan(start, end, color="#37A087", alpha=0.22)
        for alert in alerts.itertuples(index=False):
            ax.axvspan(alert.confirmed_at, alert.end, color="#DF802F", alpha=0.30)
            ax.axvline(alert.confirmed_at, color="#B65913", lw=0.8)
        for change in change_points:
            ax.axvline(change, color="#13735B", lw=0.8, linestyle=":")
        if highlight is not None:
            ax.axvline(highlight, color="#692C91", lw=1.2, linestyle="--")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.18)
    legend = [
        Patch(facecolor="#37A087", alpha=0.35, label="Labeled anomaly"),
        Patch(facecolor="#DF802F", alpha=0.45, label="Confirmed alert"),
    ] if "anomaly" in data else [Patch(facecolor="#DF802F", alpha=0.45, label="Confirmed alert")]
    if highlight is not None:
        legend.append(plt.Line2D([], [], color="#692C91", linestyle="--", label="Case of interest"))
    if len(change_points):
        legend.append(plt.Line2D([], [], color="#13735B", linestyle=":", label="Labeled change point"))
    axes[0].legend(handles=legend, loc="upper right", fontsize=8)
    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].set_xlabel("Time in recording")
    fig.suptitle(title, x=0.02, ha="left")
    fig.savefig(path, dpi=140)
    plt.close(fig)
