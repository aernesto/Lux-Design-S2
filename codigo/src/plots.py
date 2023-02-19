# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Mapping


def plot_factory_stats(agent_stats: Mapping[str, Mapping]):
    stats_df = pd.DataFrame([s['factories_total'] for s in agent_stats])
    f, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey='row')
    axix = {
        'power': [0, 0],
        'lichen': [0, 1],
        'ice': [1, 0],
        'water': [1, 1],
        'ore': [2, 0],
        'metal': [2, 1]
    }
    num_fac_allowed = stats_df.max_number_allowed.max()
    pre_phase = num_fac_allowed * 2
    for title, ix in axix.items():
        try:
            y = stats_df[title].values
        except KeyError:
            continue
        ax = axes[ix[0], ix[1]]
        ax.step(np.arange(-pre_phase,
                          len(y) - pre_phase),
                y,
                where='pre',
                marker='.')
        ax.axvline(0, color='red', alpha=.6, linestyle='dashed')
        ax.grid('on')
        ax.set_title(title)
        if title in {'ore', 'metal'}:
            ax.set_xlabel('step')
    f.tight_layout()
    return f, axes


if __name__ == "__main__":
    pass
