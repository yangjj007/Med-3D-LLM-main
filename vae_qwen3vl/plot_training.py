#!/usr/bin/env python3
"""
可视化训练进度：读取 training_metrics.jsonl，绘制 loss 与 lr 曲线。
用法:
  # 训练结束后画一张图
  python vae_qwen3vl/plot_training.py --run_dir outputs_3d_align/ep10_lr5e-5_bs4_nall_lora16_xxx

  # 动态刷新：训练的同时另开终端运行，曲线会持续更新（收敛过程实时看）
  python vae_qwen3vl/plot_training.py --run_dir outputs_3d_align/你的run目录 --live
"""

import argparse
import json
import os
import time


def _load_metrics(path):
    """Read training_metrics.jsonl, return (steps, epochs, losses, lrs) or ([], [], [], [])."""
    steps, epochs, losses, lrs = [], [], [], []
    if not path or not os.path.isfile(path):
        return steps, epochs, losses, lrs
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                steps.append(d["step"])
                epochs.append(d["epoch"])
                losses.append(d["loss"])
                lrs.append(d.get("lr", 0))
    except Exception:
        pass
    return steps, epochs, losses, lrs


def _ema_trend(losses, alpha=0.95):
    """指数移动平均，用于画趋势线：近期权重大，能看出走向而不是简单平均。"""
    if not losses:
        return []
    out = [losses[0]]
    for i in range(1, len(losses)):
        out.append(alpha * out[-1] + (1 - alpha) * losses[i])
    return out


def _draw(ax1, ax2, steps, losses, lrs, smooth_ratio, title_suffix="", fig=None, trend_alpha=0.95, log_x=False):
    """Update axes with current data. Returns current trend value (EMA at last step) or None."""
    ax1.clear()
    ax2.clear()
    if log_x and steps:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    if not steps:
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Learning rate")
        ax2.set_xlabel("Step")
        if fig is not None:
            fig.tight_layout()
        return None
    x_plot = [max(1, s) for s in steps] if log_x else steps  # log 轴时 step=0 会出错
    ax1.plot(x_plot, losses, alpha=0.3, color="C0", label="loss (raw)")
    # 趋势线：EMA，用于透过波动看下降趋势
    trend = _ema_trend(losses, alpha=trend_alpha)
    ax1.plot(x_plot, trend, color="C2", linewidth=2, label="趋势 (EMA)")
    current_trend = trend[-1] if trend else None
    if smooth_ratio > 0 and len(losses) > 1:
        from collections import deque
        w = max(1, int(len(losses) * smooth_ratio))
        smoothed = []
        q = deque(losses[:w], maxlen=w)
        for i in range(len(losses)):
            if i >= w:
                q.append(losses[i])
            smoothed.append(sum(q) / len(q))
        ax1.plot(x_plot, smoothed, color="C0", linewidth=1.5, alpha=0.7, label="loss (smoothed)")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    if title_suffix:
        ax1.set_title(f"Training {title_suffix}")

    ax2.plot(x_plot, lrs, color="C1", alpha=0.8)
    ax2.set_ylabel("Learning rate")
    ax2.set_xlabel("Step")
    ax2.grid(True, alpha=0.3)
    if fig is not None:
        fig.tight_layout()
    return current_trend


def main():
    parser = argparse.ArgumentParser(description="Plot training loss and LR from training_metrics.jsonl")
    parser.add_argument("--metrics", type=str, default=None, help="Path to training_metrics.jsonl")
    parser.add_argument("--run_dir", type=str, default=None, help="Run directory (metrics file: run_dir/training_metrics.jsonl)")
    parser.add_argument("--out", type=str, default=None, help="Output image path (default: same dir as metrics, training_curve.png)")
    parser.add_argument("--smooth", type=float, default=0, help="Smoothing window for loss (0=no smooth, e.g. 0.1 for 10%%)")
    parser.add_argument("--trend_alpha", type=float, default=0.95, help="EMA alpha for trend line (0~1, 越大越平滑，默认0.95)")
    parser.add_argument("--log_x", action="store_true", help="横轴用对数刻度，步数多时能看清前期下降趋势")
    parser.add_argument("--live", action="store_true", help="Live mode: refresh plot periodically while training runs (run in another terminal)")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds when --live (default: 5)")
    args = parser.parse_args()

    if args.run_dir:
        path = os.path.join(args.run_dir, "training_metrics.jsonl")
    else:
        path = args.metrics
    if not path:
        print("Provide --metrics or --run_dir")
        return 1

    try:
        import matplotlib
        if args.live:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                matplotlib.use("Agg")
        else:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to plot: pip install matplotlib")
        return 1

    headless_live = False
    if args.live:
        # 无 DISPLAY 时直接用 Agg，避免 TkAgg 在首次创建 figure 时报错
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
            headless_live = True
            print("(无图形界面，--live 将定期更新 training_curve.png，可用 watch 或刷新查看)")
        else:
            # 有 DISPLAY 时尝试弹窗；若首次创建 figure 失败（如 headless 检测），则回退 Agg
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
                plt.close(fig)
            except Exception:
                matplotlib.use("Agg")
                headless_live = True
                print("(图形后端不可用，--live 将定期更新 training_curve.png)")

    if args.live:
        # 动态刷新：循环读文件并重绘
        if not os.path.isdir(os.path.dirname(path)):
            print(f"Run dir not found yet: {os.path.dirname(path)}. Create it by starting training first.")
            return 1
        if not headless_live:
            plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        if not headless_live and hasattr(fig.canvas, "manager") and getattr(fig.canvas.manager, "set_window_title", None):
            try:
                fig.canvas.manager.set_window_title("Training progress (live)")
            except Exception:
                pass
        last_n = 0
        out_path = args.out or os.path.join(os.path.dirname(path), "training_curve.png")
        try:
            while True:
                steps, epochs, losses, lrs = _load_metrics(path)
                if steps:
                    title_suffix = f"| steps={len(steps)}"
                    trend_val = _draw(ax1, ax2, steps, losses, lrs, args.smooth, title_suffix, fig=fig, trend_alpha=args.trend_alpha, log_x=args.log_x)
                    if trend_val is not None:
                        print(f"\r  趋势线当前值: {trend_val:.6f}  (step={steps[-1]})    ", end="", flush=True)
                    if not headless_live:
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    if len(steps) != last_n:
                        last_n = len(steps)
                    fig.savefig(out_path, dpi=150)
                else:
                    ax1.clear()
                    ax1.set_title("Waiting for training_metrics.jsonl... (start training in another terminal)")
                    ax1.set_ylabel("Loss")
                    ax2.clear()
                    ax2.set_ylabel("Learning rate")
                    ax2.set_xlabel("Step")
                    fig.tight_layout()
                    if not headless_live:
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                    fig.savefig(out_path, dpi=150)
                if headless_live:
                    time.sleep(args.interval)
                else:
                    plt.pause(args.interval)
        except KeyboardInterrupt:
            print("Live plot stopped.")
        if not headless_live:
            plt.ioff()
            plt.show()
        return 0

    # 非 live：训练结束后画一张图
    if not os.path.isfile(path):
        print(f"Metrics file not found: {path}")
        return 1
    steps, epochs, losses, lrs = _load_metrics(path)
    if not steps:
        print("No data in metrics file.")
        return 1

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    trend_val = _draw(ax1, ax2, steps, losses, lrs, args.smooth, fig=fig, trend_alpha=args.trend_alpha, log_x=args.log_x)
    out_path = args.out or os.path.join(os.path.dirname(path), "training_curve.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    if trend_val is not None:
        print(f"趋势线当前值: {trend_val:.6f}  (step={steps[-1]})")
    return 0


if __name__ == "__main__":
    exit(main())
