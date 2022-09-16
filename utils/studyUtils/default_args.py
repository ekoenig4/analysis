kstest = dict(
    empirical=True,
    e_show=False, 
    e_difference=True,
    e_d_legend_frameon=True
)

datamc = dict(
    ratio=True,
    r_ylabel="Data/MC",
    r_ylim=(0.65,1.35)
)

auroc = dict(
    empirical=True,
    e_show=False,
    e_correlation=True,
    e_c_method=None,
    e_c_label_stat='area',
    e_c_legend_frameon=True,
)