"""Plot saved, paired episodes without rerunning or changing either controller."""
import argparse
from pathlib import Path

import numpy as np

from .artifacts import read_json


def plot(paths, episode, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if episode < 0:
        raise ValueError('episode must be nonnegative')
    records = [(Path(p), read_json(p)) for p in paths]
    rows = [r['episodes'][episode] for _, r in records]
    if len({row['mission_id'] for row in rows}) != 1:
        raise ValueError('Select identical saved missions for trajectory comparisons')
    if len({r['plant'] for _, r in records}) != 1:
        raise ValueError('Compare controllers on the same plant; plot transfer cases separately')
    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
    ax, errors, speeds = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    colors = plt.get_cmap('tab10')
    styles = ['-', '--', ':', '-.']
    labels = []
    for m, ((path, result), row) in enumerate(zip(records, rows)):
        with np.load(path.parent / row['trajectory_file'], allow_pickle=False) as archive:
            time, states, goals, deadline = (archive[k] for k in ('time', 'states', 'goals', 'deadline'))
        method = result['method']
        labels.append(f"{method} s{result['training_seed']}: " +
                      ('success' if row['success'] else 'collision' if row['collision'] else
                       'timeout' if row['timeout'] else 'timing missed'))
        for i in range(states.shape[1]):
            options = dict(color=colors(i % 10), linestyle=styles[m % 4])
            ax.plot(states[:, i, 0], states[:, i, 1], **options, label=f'{method}, car {i+1}')
            if m == 0:
                ax.scatter(*states[0, i, :2], color=colors(i % 10), marker='o', s=22)
                ax.scatter(*goals[i], color=colors(i % 10), marker='x', s=65)
            errors.plot(time, np.linalg.norm(states[:, i, :2] - goals[i], axis=1), **options)
            speeds.plot(time, states[:, i, 3], **options)
        if m == 0:
            for axis in (errors, speeds):
                axis.axvline(float(deadline), color='black', linestyle=':', label='deadline')
    ax.set(aspect='equal', xlabel='x (m)', ylabel='y (m)', title='; '.join(labels))
    ax.legend(fontsize=7, loc='best', ncol=min(len(records), 3))
    errors.set(xlabel='Time (s)', ylabel='Distance to own goal (m)')
    speeds.set(xlabel='Time (s)', ylabel='Forward speed (m/s)')
    fig.suptitle(f"Saved mission {rows[0]['mission_id'][:12]} — {records[0][1]['plant']}; traces sampled at policy decisions", fontsize=10)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + extension), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, nargs='+', required=True)
    p.add_argument('--episode', type=int, default=0, help='zero-based index in paired manifests')
    p.add_argument('--output', type=Path, required=True, help='filename stem for PNG/PDF')
    args = p.parse_args()
    plot(args.results, args.episode, args.output)


if __name__ == '__main__':
    main()
