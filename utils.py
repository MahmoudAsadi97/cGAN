"""Results analysis (reference: cGAN_EB ``utils.py``), adapted to the 64 Fluvius labels.

Portions adapted from cGAN_EB, Copyright (c) 2024 malvin0130, MIT License
(see ``LICENSE-cGAN_EB``).

``real_results`` / ``generate_results`` write one CSV per label (``<label>_real.csv``,
``<label>_gen.csv``) as the reference does; ``stats_plot`` reproduces the
validation figures of Li, Dong & Qiu (2025):

    temp_Loads_weekday/weekend.png   average real vs generated day per label (Figs. 3-4)
    temp_plot[_code]_weekday/weekend.png   difference in mean and std of the key parameters (Figs. 5-6)
    temp_peak_example[_code].png / temp_base_example[_code].png   near-peak / near-base
                                     distributions (Figs. 7-8)
    temp_KLD[_code]_weekday/weekend.png    KL divergence of the key parameters (Figs. 9-10)
    temp_FID[_code]_weekday/weekend.png    FID of the key parameters (Figs. 11-12)
    temp_dist_<Key>[_code].png       distribution of one key parameter in all 64 labels
                                     (8 groups x 8 day-type/season columns), real vs generated
    stat_dif_plot[_code].csv, kld[_code].csv, fid[_code].csv   the numbers behind the heatmaps

``definitions="paper"`` (no suffix) uses the key parameters as defined in the
paper's text (``metrics.key_parameters``: 97.5th / 2.5th percentiles, Price's
rise and fall time) and a KL divergence between the two distributions
(``metrics.kl_divergence``).  ``definitions="code"`` (suffix ``_code``) uses
exactly what the cGAN_EB code computes and what the published figures show:
the 2nd-highest / 2nd-lowest hour, index differences for rise and fall
(``metrics.reference_key_parameters``) and its "KLD", which pairs real day i
with generated day i (``metrics.reference_kld``).  The FID per key parameter is
the 1-D Frechet distance in both, as in ``compute_fid``.

Further differences to the reference: "real" are the days of held-out *test*
households (the reference compares with its own training data); the example
labels of Figs. 7-8 are, as in the paper, the weekday and the weekend label
with the largest (left) and smallest (right) difference in mean, and can be
fixed so that two models are shown on the same labels; figures are written
with the Agg backend.  ``compare_profiles`` draws Figs. 3-4 with several models
in each panel.
"""
import os

import matplotlib
import numpy as np
import pandas as pd

from data_prep import DAYTYPE_NAMES, GROUP_NAMES, SEASON_NAMES, label_names
from metrics import PAPER_KEYS, key_parameters, kl_divergence, reference_key_parameters, reference_kld

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

cats_ = label_names()
cats_1 = [f"{g} {s}" for g in GROUP_NAMES for s in SEASON_NAMES]
KEY_TITLES = {"near_peak": "PeakLoad", "near_base": "BaseLoad", "peak_duration": "PeakDuration",
              "rise_time": "RiseTime", "fall_time": "FallTime"}
# Key parameters of the cGAN_EB code, in the same order and with the same titles.
CODE_KEYS = {"Peak_load": "PeakLoad", "Base_load": "BaseLoad", "Peak_duration": "PeakDuration",
             "Rise": "RiseTime", "Fall": "FallTime"}
COLORS = {"Real": "#F24405", "Generated": "#008F8C"}


def real_results(split, folder):
    os.makedirs(folder, exist_ok=True)
    for cat_num, cat in enumerate(cats_):
        df_temp = pd.DataFrame(split.X[split.label == cat_num])
        df_temp.to_csv(os.path.join(folder, cat + '_real.csv'), index=False, header=True)


def generate_results(wrapped, num_samples, folder, seed=0):
    """``num_samples`` generated days per label (an int, or a dict label -> count)."""
    os.makedirs(folder, exist_ok=True)
    for cat_num, cat in enumerate(cats_):
        n = num_samples[cat_num] if isinstance(num_samples, dict) else num_samples
        df_temp = pd.DataFrame(wrapped.sample_kwh(np.full(n, cat_num), seed + cat_num))
        df_temp.to_csv(os.path.join(folder, cat + '_gen.csv'), index=False, header=True, float_format='%.4f')


def stat_df_dict(folder):
    df_gens, df_reals = {}, {}
    for label in cats_:
        df_gens[label] = pd.read_csv(os.path.join(folder, f'{label}_gen.csv'))
        df_reals[label] = pd.read_csv(os.path.join(folder, f'{label}_real.csv'))
    return df_gens, df_reals


def _split_heatmaps(df, folder, stem, **kw):
    """Heatmap of all labels plus separate weekday / weekend heatmaps (as the reference)."""
    sns.set(font_scale=1.4)
    plt.rcParams['figure.figsize'] = [18, 9]
    cm = sns.heatmap(df, **kw)
    plt.tight_layout()
    cm.figure.savefig(os.path.join(folder, f'{stem}.png'))
    plt.close()
    for d, name in enumerate(DAYTYPE_NAMES):
        part = df.iloc[:, 32 * d:32 * (d + 1)].copy()
        part.columns = cats_1
        cm = sns.heatmap(part, **kw)
        plt.tight_layout()
        cm.figure.savefig(os.path.join(folder, f'{stem}_{name.lower()}.png'))
        plt.close()


def _profile_grid(df_reals, gens, folder, daytype, stem='temp_Loads'):
    """Average day per label of the real data and of one or more generators (``gens``: name -> dfs)."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = [16, 24]
    figure, ax = plt.subplots(8, 4, sharex=True, sharey=True)
    styles = [dict(color='#008F8C', lw=5), dict(color='#3B5BA9', lw=3, ls='--'), dict(color='#7E22CE', lw=3, ls=':')]
    for label_num in range(32):
        label = cats_[32 * daytype + label_num]
        row_idx, col_idx = label_num // 4, label_num % 4
        ax[row_idx, col_idx].plot(df_reals[label].mean().to_numpy(), color=COLORS['Real'], lw=5, label='Real')
        for (name, df_gens), style in zip(gens.items(), styles):
            ax[row_idx, col_idx].plot(df_gens[label].mean().to_numpy(), label=name, **style)
        if col_idx == 0:
            ax[row_idx, col_idx].set_ylabel(f'{GROUP_NAMES[row_idx]}\n\nLoad (kWh)', fontsize=14)
        ax[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=14)
    for s_, season in enumerate(SEASON_NAMES):
        ax[0, s_].set_title(season, fontsize=16)
        ax[7, s_].set_xlabel('Time of day', fontsize=14)
    ax[7, 0].set_xticks(np.arange(0, 25, 3), ['0', '3', '6', '9', '12', '15', '18', '21', '24'])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(labels), fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    figure.savefig(os.path.join(folder, f'{stem}_{DAYTYPE_NAMES[daytype].lower()}.png'))
    plt.close(figure)


def _distribution_examples(gen_kp, real_kp, labels, key, xlabel, path):
    plt.rcParams['figure.figsize'] = [18, 9]
    fig, ax = plt.subplots(2, 2, sharey=True)
    for i, label in enumerate(labels):
        r, g = real_kp[label][key], gen_kp[label][key]
        a = ax[i // 2, i % 2]
        bins = np.histogram(np.hstack((r, g)), bins=15)[1]
        sns.set_style('white')
        sns.histplot(r, color=COLORS['Real'], label='Real', stat='percent', bins=bins, kde=True,
                     line_kws={'lw': 5}, ax=a)
        sns.histplot(g, color=COLORS['Generated'], label='Generated', stat='percent', bins=bins, kde=True,
                     line_kws={'lw': 5}, ax=a)
        a.set_title(label, fontsize=18)
        a.set_xlabel(xlabel, fontsize=16)
        if i % 2 == 0:
            a.set_ylabel('Frequency (%)', fontsize=16)
        a.tick_params(axis='both', which='major', labelsize=16)
    handles, labs = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    fig.savefig(path)
    plt.close(fig)


def example_labels(dif: pd.DataFrame, title: str) -> list[str]:
    """As in the paper (Figs. 7-8): per day type the label with the largest (left) and smallest (right)
    absolute difference in mean -> [weekday large, weekday small, weekend large, weekend small]."""
    out = []
    for d in range(2):
        diff = dif.loc[f'{title}_mean'].iloc[32 * d:32 * (d + 1)].abs().sort_values()
        out += [diff.index[-1], diff.index[0]]
    return out


KEY_NAMES = {"PeakLoad": ("Near-peak load", "kWh"), "BaseLoad": ("Near-base load", "kWh"),
             "PeakDuration": ("Peak load duration", "h"), "RiseTime": ("Rise time", "h"),
             "FallTime": ("Fall time", "h")}


def distribution_grid(gen_kp, real_kp, key, title, path):
    """Distribution of one key parameter, real vs generated, in all 64 labels: rows = the 8 technology
    groups, columns = the 4 seasons of weekdays and then of weekends.  Each panel gives the
    Kolmogorov-Smirnov distance between the two samples."""
    from metrics import _ks

    name, unit = KEY_NAMES[title]
    plt.style.use('default')
    fig, ax = plt.subplots(8, 8, figsize=(24, 22))
    for d in range(2):
        for g in range(8):
            for s in range(4):
                label = cats_[32 * d + 4 * g + s]
                r, gen = real_kp[label][key], gen_kp[label][key]
                a = ax[g, 4 * d + s]
                discrete = np.allclose(r, np.round(r)) and np.allclose(gen, np.round(gen))
                if discrete:
                    lo, hi = min(r.min(), gen.min()), max(r.max(), gen.max())
                    bins = np.arange(lo - 0.5, hi + 1.5)
                else:  # 0.5-99.5 % of both samples; values outside are counted in the edge bins
                    lo, hi = np.percentile(np.r_[r, gen], [0.5, 99.5])
                    bins = np.linspace(lo, hi, 26)
                clip = lambda v: np.clip(v, bins[0] + 1e-9, bins[-1] - 1e-9)
                wr = np.full(len(r), 100 / len(r))
                wg = np.full(len(gen), 100 / len(gen))
                a.hist(clip(r), bins, weights=wr, color=COLORS['Real'], alpha=0.35, label='Real')
                a.hist(clip(r), bins, weights=wr, histtype='step', color=COLORS['Real'], lw=1.2)
                a.hist(clip(gen), bins, weights=wg, histtype='step', color=COLORS['Generated'], lw=1.8,
                       label='Generated')
                a.text(0.97, 0.93, f"KS {_ks(r, gen):.2f}", transform=a.transAxes, ha='right', va='top', fontsize=9)
                a.tick_params(labelsize=8)
                if g == 0:
                    a.set_title(f"{DAYTYPE_NAMES[d]}\n{SEASON_NAMES[s]}", fontsize=12)
                if s == 0 and d == 0:
                    a.set_ylabel(f"{GROUP_NAMES[g]}\n\n%", fontsize=11)
                if g == 7:
                    a.set_xlabel(f"{name} ({unit})", fontsize=9)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=14)
    fig.suptitle(f"{name} ({unit}): real vs generated days in all 64 labels", fontsize=18)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def stats_plot(folder, definitions='paper', examples=None):
    """Figs. 3-12 for the CSVs in ``folder``.  ``examples``: {"near_peak": [4 labels], "near_base": [...]}
    to show fixed labels in Figs. 7-8; returns the tables and the example labels used."""
    df_gens, df_reals = stat_df_dict(folder)
    if definitions == 'paper':
        kp_fn, keys, suffix = key_parameters, {k: KEY_TITLES[k] for k in PAPER_KEYS}, ''
    elif definitions == 'code':
        kp_fn, keys, suffix = reference_key_parameters, CODE_KEYS, '_code'
    else:
        raise ValueError("definitions must be 'paper' or 'code'")
    gen_kp = {lab: kp_fn(df_gens[lab].to_numpy()) for lab in cats_}
    real_kp = {lab: kp_fn(df_reals[lab].to_numpy()) for lab in cats_}

    # Differences of mean and std (generated - real), Figs. 5-6
    rows = {}
    for label in cats_:
        col = [gen_kp[label][k].mean() - real_kp[label][k].mean() for k in keys]
        col += [gen_kp[label][k].std(ddof=1) - real_kp[label][k].std(ddof=1) for k in keys]
        rows[label] = col
    index = [f'{t}_mean' for t in keys.values()] + [f'{t}_std' for t in keys.values()]
    dif = pd.DataFrame(rows, index=index)
    dif.to_csv(os.path.join(folder, f'stat_dif_plot{suffix}.csv'), index=True, header=True)
    order = [i for t in keys.values() for i in (f'{t}_mean', f'{t}_std')]
    _split_heatmaps(dif.loc[order], folder, f'temp_plot{suffix}', cmap='vlag', vmin=-1.5, vmax=1.5)

    # KL divergence, Figs. 9-10
    if definitions == 'paper':
        kld = pd.DataFrame({label: [kl_divergence(real_kp[label][k], gen_kp[label][k]) for k in keys]
                            for label in cats_}, index=list(keys.values()))
    else:
        kld = pd.DataFrame({label: list(reference_kld(real_kp[label], gen_kp[label]).values()) for label in cats_},
                           index=list(keys.values()))
    kld.to_csv(os.path.join(folder, f'kld{suffix}.csv'))
    _split_heatmaps(kld, folder, f'temp_KLD{suffix}', cmap='Blues', vmin=0, vmax=1)

    # FID of each key parameter (1-D Frechet distance), Figs. 11-12
    fid = pd.DataFrame({label: [(real_kp[label][k].mean() - gen_kp[label][k].mean()) ** 2
                                + (real_kp[label][k].std(ddof=1) - gen_kp[label][k].std(ddof=1)) ** 2
                                for k in keys] for label in cats_}, index=list(keys.values()))
    fid.to_csv(os.path.join(folder, f'fid{suffix}.csv'))
    _split_heatmaps(fid, folder, f'temp_FID{suffix}', cmap='cubehelix_r', vmin=0, vmax=10, linewidths=1.2)

    # Average days, Figs. 3-4 (identical for both definitions)
    if definitions == 'paper':
        for d in range(2):
            _profile_grid(df_reals, {'Generated': df_gens}, folder, d)

    # Near-peak / near-base distributions, Figs. 7-8
    peak_key, base_key = list(keys)[0], list(keys)[1]
    examples = examples or {}
    used = {}
    for key, name, xlabel, stem in ((peak_key, 'near_peak', 'Peak load', 'temp_peak_example'),
                                    (base_key, 'near_base', 'Base load', 'temp_base_example')):
        labels = examples.get(name) or example_labels(dif, keys[key])
        used[name] = labels
        _distribution_examples(gen_kp, real_kp, labels, key, xlabel, os.path.join(folder, f'{stem}{suffix}.png'))

    # Every key parameter in all 64 labels (extends Figs. 7-8 from four example labels to all of them)
    for key, title in keys.items():
        distribution_grid(gen_kp, real_kp, key, title, os.path.join(folder, f'temp_dist_{title}{suffix}.png'))
    return {'dif': dif, 'kld': kld, 'fid': fid, 'examples': used}


def compare_profiles(folders: dict, out_folder):
    """Figs. 3-4 with several generators per panel; ``folders``: legend name -> folder with *_gen.csv
    (the real days are read from the first folder)."""
    os.makedirs(out_folder, exist_ok=True)
    first = next(iter(folders.values()))
    df_reals = {lab: pd.read_csv(os.path.join(first, f'{lab}_real.csv')) for lab in cats_}
    gens = {name: {lab: pd.read_csv(os.path.join(f, f'{lab}_gen.csv')) for lab in cats_}
            for name, f in folders.items()}
    for d in range(2):
        _profile_grid(df_reals, gens, out_folder, d, stem='compare_Loads')
