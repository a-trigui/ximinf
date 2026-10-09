from astropy.cosmology import Planck18
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from getdist import plots, MCSamples

BLUE = '#1F487E'
RED = '#A31621'
GOLD = '#C9A227'
BEIGE = 'beige'

GREEN = '#687444'
PURPLE = '#5E4983'

styles = {
    "NRE":        {"color": RED, "filled": True},
    "Standax":    {"color": GOLD, "filled": False},
    "MLE": {"color": BLUE, "filled": False},
}

def apply_default_settings():
    plt.rcParams.update({
    "font.size": 12,          # General font size
    "axes.labelsize": 14,     # x/y axis labels
    "axes.titlesize": 16,     # Title
    "xtick.labelsize": 12,    # x tick labels
    "ytick.labelsize": 12,    # y tick labels
    "legend.fontsize": 12,    # Legend
})

def plot_HD(
    example,
    params,
    default_params,
    N_total,
    survey_name=None,
    images_dir=None,
    n_bins=15,
):
    # ============================================
    # Simulation parameters
    # ============================================

    mabs = params['mabs']
    beta = params['beta']
    alpha = params['alpha']

    # ============================================
    # Data from simulation
    # ============================================

    z = np.asarray(example['z'])
    magobs = np.asarray(example['magobs'])
    c = np.asarray(example['c'])

    # ============================================
    # Apply selection
    # ============================================

    mask = magobs > 0

    z = z[mask]
    magobs = magobs[mask]
    c = c[mask]

    # ============================================
    # Number of SNe
    # ============================================

    N_sne_selected = len(z)

    # If N_total is the number before selection
    N_sne = N_total

    # ============================================
    # Fiducial cosmology distance modulus
    # ============================================

    mu_th = Planck18.distmod(z).value

    # ============================================
    # Raw Hubble residuals
    # ============================================

    residuals = magobs - mu_th - mabs

    # ============================================
    # Equal-count (quantile) binning
    # ============================================

    mask_finite = (
        np.isfinite(z)
        & np.isfinite(residuals)
    )

    z_clean = z[mask_finite]
    res_clean = residuals[mask_finite]

    # Quantile bins
    bins = np.quantile(
        z_clean,
        np.linspace(0, 1, n_bins + 1)
    )

    bins = np.unique(bins)

    bin_idx = np.digitize(z_clean, bins) - 1

    z_bin = []
    res_bin = []
    res_err = []

    for i in range(len(bins) - 1):

        mask_bin = bin_idx == i

        if np.sum(mask_bin) == 0:
            continue

        z_i = z_clean[mask_bin]
        r_i = res_clean[mask_bin]

        z_bin.append(np.mean(z_i))
        res_bin.append(np.mean(r_i))
        res_err.append(np.std(r_i) / np.sqrt(len(r_i)))

    z_bin = np.array(z_bin)
    res_bin = np.array(res_bin)
    res_err = np.array(res_err)

    # ============================================
    # Colormap
    # ============================================

    cmap = LinearSegmentedColormap.from_list(
        'custom_red_beige_blue',
        [RED, BEIGE, BLUE]
    )

    norm = TwoSlopeNorm(
        vmin=-0.2,
        vcenter=0.0,
        vmax=0.3
    )

    # ============================================
    # Figure with two panels
    # ============================================

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(6, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # ============================================
    # TOP PANEL: observed magnitudes
    # ============================================

    scatter = ax1.scatter(
        z,
        magobs - mabs,
        c=c,
        cmap=cmap,
        norm=norm,
        edgecolor='none',
        alpha=0.8,
    )

    # ============================================
    # Fiducial cosmology curve
    # ============================================

    z_plot = np.linspace(
        z.min(),
        z.max(),
        1000
    )

    mu_plot = Planck18.distmod(z_plot).value

    ax1.plot(
        z_plot,
        mu_plot,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Planck18'
    )

    # ============================================
    # Low-z selection cut
    # ============================================

    if survey_name is not None:

        if f'cut_loc_{survey_name}' not in params:
            cut_loc = default_params[f'cut_loc_{survey_name}']
        else:
            cut_loc = params[f'cut_loc_{survey_name}']

        # Convert observed-magnitude cut to the quantity
        # plotted on the y-axis: magobs - mabs
        ax1.hlines(
            cut_loc - mabs,
            xmin=z.min(),
            xmax=z.max(),
            colors='grey',
            linestyles='dashed',
            linewidth=1.5,
            label='low-$z$ cut'
        )

    ax1.legend()

    # ============================================
    # Histogram above top panel
    # ============================================

    divider = make_axes_locatable(ax1)

    ax_hist = divider.append_axes(
        "top",
        1.2,
        pad=0.0,
        sharex=ax1
    )

    ax_hist.hist(
        z,
        bins=20,
        color='gray',
        edgecolor='black'
    )

    ax_hist.axis('off')

    # ============================================
    # Inset histogram of colour parameter c
    # ============================================

    color_clean_hist = c[np.isfinite(c)]

    ax_inset = ax1.inset_axes(
        [0.55, 0.2, 0.35, 0.25]
    )

    counts, bins_hist = np.histogram(
        color_clean_hist,
        bins=20
    )

    bin_centers = 0.5 * (
        bins_hist[:-1] + bins_hist[1:]
    )

    bin_colors = cmap(norm(bin_centers))

    ax_inset.bar(
        bins_hist[:-1],
        counts,
        width=bins_hist[1:] - bins_hist[:-1],
        color=bin_colors,
        edgecolor='black'
    )

    ax_inset.set_xlabel(r'$c$')
    ax_inset.set_yticks([])
    ax_inset.set_frame_on(False)

    # ============================================
    # BOTTOM PANEL: residuals
    # ============================================

    ax2.scatter(
        z,
        residuals,
        c=c,
        cmap=cmap,
        norm=norm,
        edgecolor='none',
        alpha=0.4,
        s=10
    )

    # ============================================
    # Binned residuals
    # ============================================

    ax2.errorbar(
        z_bin,
        res_bin,
        yerr=res_err,
        fmt='o',
        color='black',
        markersize=5,
        capsize=3
    )

    # ============================================
    # Reference line
    # ============================================

    ax2.axhline(
        0.0,
        color='black',
        linestyle='--',
        linewidth=1
    )

    # ============================================
    # Labels
    # ============================================

    ax1.set_ylabel(r'$\mu$')
    ax2.set_xlabel(r'Redshift $z$')
    ax2.set_ylabel(r'$\Delta\mu$')

    # ============================================
    # Figure title
    # ============================================

    fig.suptitle(
        rf'$M_{{\rm abs}} = {mabs:.3f}$, '
        rf'$\beta = {beta:.3f}$, '
        rf'$\alpha = {alpha:.3f}$'
        '\n'
        rf'$N_{{\rm SNe}} = {N_sne}$, '
        rf'$N_{{\rm selected}} = {N_sne_selected}$',
        fontsize=12
    )

    # ============================================
    # Layout
    # ============================================

    plt.tight_layout()

    plt.subplots_adjust(
        hspace=0.05,
        top=0.90
    )

    # ============================================
    # Save figure
    # ============================================

    if images_dir is not None:
        plt.savefig(
            images_dir / f"sim_example.png",
            bbox_inches='tight'
        )

    plt.show()

    # ============================================
    # Diagnostics
    # ============================================

    print('Mean residual:', np.mean(residuals))
    print('Residual scatter:', np.std(residuals))

    return fig, (ax1, ax2)




def plot_residuals(data_filt, params, global_param_names, mask):
    cmap2 = LinearSegmentedColormap.from_list(
        'custom_green_beige_purple',
        [GREEN, BEIGE, PURPLE]
    )

    cmap3 = LinearSegmentedColormap.from_list(
        'custom_blue_beige_orange',
        [RED, BEIGE, BLUE]
    )

    # Create figure and horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    
    # First subplot: z vs magobs
    axes[0].axhline(0, c='gray', ls='--', zorder=0)
    sc1 = axes[0].scatter( #errorbar
        data_filt['z'],
        data_filt['magobs'],
        # data_filt['magobs_err'],
        # fmt='o',
        color='gray',
        edgecolor='k',
        # markeredgecolor='k',
        # markerfacecolor='gray',
        alpha=0.7,
    )
    axes[0].set_title('Magnitude vs Redshift', fontsize=14)
    axes[0].set_xlabel('Redshift (z)', fontsize=12)
    axes[0].set_ylabel('Observed Magnitude', fontsize=12)
    
    
    # Second subplot: c vs magobs
    axes[1].axhline(0, c='gray', ls='--', zorder=0)
    sc2 = axes[1].scatter(
        data_filt['c'],
        data_filt['magobs'],
        c=data_filt['x1'],
        cmap=cmap2,
        edgecolor='k'
    )
    axes[1].set_title('Magnitude vs Color', fontsize=14)
    axes[1].set_xlabel('Color (c)', fontsize=12)
    axes[1].set_ylabel('Observed Magnitude', fontsize=12)
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label('Stretch x1', fontsize=12)

    # Third subplot: x1 vs magobs
    axes[2].axhline(0, c='gray', ls='--', zorder=0)
    sc3 = axes[2].scatter(
        data_filt['x1'],
        data_filt['magobs'],
        c=data_filt['c'],
        cmap=cmap3,
        edgecolor='k'
    )
    axes[2].set_title('Magnitude vs Stretch', fontsize=14)
    axes[2].set_xlabel('Stretch (x1)', fontsize=12)
    axes[2].set_ylabel('Observed Magnitude', fontsize=12)
    cbar3 = plt.colorbar(sc3, ax=axes[2])
    cbar3.set_label('Colour c', fontsize=12)

    # Construct the title string dynamically
    title_str = ", ".join(
        f"{name} = {value:.2f}"
        for name, value in params.items()
        if name in global_param_names
    )

    fig.suptitle(title_str, fontsize=16)

    print(f"{sum(mask)} supernovae")

    plt.show()


def plot_corner_comparison(
    posterior_dicts,
    truth_dict=None,
    styles=styles,
    methods_to_plot=("NRE", "MLE"),
    contours=(0.68, 0.95),
    latex_labels=None,
    save_path="./Images/corner_comparison.png",
    legend_loc="upper right",
    show=True,
):
    """
    Corner plot comparing posteriors from several inference methods.

    Parameters
    ----------
    posterior_dicts : dict[str, dict[str, array-like]]
        {method_name: {param_name: samples}}.
    truth_dict : dict[str, float], optional
        True parameter values, drawn as markers on the parameters shared
        by the selected methods.
    styles : dict[str, dict], optional
        {method_name: {"color": ..., "filled": bool}}. Defaults to
        matplotlib's color cycle with unfilled contours when omitted.
    methods_to_plot : sequence of str
        Methods to display (order = plot order),
        e.g. ["SBI", "Standax", "cosmologix"].
    contours : sequence of float
        Credible levels of the contours.
    save_path : str or None
        Where to export the figure. None to skip saving.
    legend_loc : str
        Legend location.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    g : getdist.plots.GetDistPlotter
    """
    plt.close()

    methods_to_plot = list(methods_to_plot)
    truth_dict = truth_dict or {}

    # Sanity check
    unknown = [m for m in methods_to_plot if m not in posterior_dicts]
    assert not unknown, (
        f"Unknown method(s): {unknown}. Available: {list(posterior_dicts)}"
    )

    # Default styles
    if styles is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        styles = {
            m: {"color": cycle[i % len(cycle)], "filled": False}
            for i, m in enumerate(methods_to_plot)
        }

    # Parameters common to the selected methods only
    common_names = sorted(
        set.intersection(*[set(posterior_dicts[m]) for m in methods_to_plot])
    )
    assert common_names, "The selected methods share no common parameters."
    latex_labels = latex_labels or {}
    labels_common = [
        latex_labels.get(n, n.replace("_", r"\_"))   # fallback: escaped raw name
        for n in common_names
    ]
    markers = {k: v for k, v in truth_dict.items() if k in common_names}

    # Build one MCSamples per selected method
    gd_samples = []
    for m in methods_to_plot:
        samples = np.column_stack([posterior_dicts[m][n] for n in common_names])
        gd = MCSamples(samples=samples, names=common_names, labels=labels_common)
        gd.updateSettings({"contours": list(contours)})
        gd_samples.append(gd)

    # Plot
    g = plots.get_subplot_plotter()
    g.settings.legend_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.axes_fontsize = 16
    g.settings.title_limit_fontsize = 14
    g.settings.title_limit_labels = False

    g.triangle_plot(
        gd_samples,
        filled=[styles[m]["filled"] for m in methods_to_plot],
        contour_colors=[styles[m]["color"] for m in methods_to_plot],
        contour_args={"lw": 1.0},
        legend_labels=methods_to_plot,
        legend_loc=legend_loc,
        markers=markers,
        title_limit=1,
    )

    if save_path is not None:
        g.export(save_path)
    if show:
        plt.show()

    return g