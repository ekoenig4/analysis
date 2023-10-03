kstest = dict(
    empirical=True,
    e_show=False, 
    e_difference=True,
    e_d_legend_frameon=True
)

datamc = dict(
    ratio=True,
    r_ylabel="Data/MC",
)

auroc = dict(
    empirical=True,
    e_show=False,
    e_correlation=True,
    e_c_method='roc',
    e_c_label_stat='area',
    e_c_legend_frameon=True,
    e_c_ylabel='Sig Eff',
    e_c_xlabel='Bkg Eff'
)