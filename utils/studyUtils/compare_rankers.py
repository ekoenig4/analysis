def get_studies(signals, bkgs, modules):
  use_bkg = any((any(bkg) for bkg in bkgs))
  h_linestyle = [':','--','-'] if len(modules) == 3 else [':','-']
  def compare_modules(var, bkg=use_bkg, figax=None, **kwargs):
    n = len(signals[0])+1 if bkg else len(signals[0])
    if figax is None:
      figax = study.get_figax(n, dim=(-1,n))
    fig, axs = figax
    
    label = list(modules.keys())

    for i, samples in enumerate(zip(*signals)):
      study.quick(
        list(samples), legend=True,
        label=label,
        h_linestyle=h_linestyle,
        varlist=[var],
        text=(0.0,1.0, samples[0].sample),
        text_style=dict(ha='left',va='bottom'),
        figax=(fig,axs.flat[i]),
        **kwargs,
      )

    if not bkg: return

    study.quick_region(
      *bkgs, legend=True,
      h_color=['grey']*3,
      label=label,
      h_linestyle=h_linestyle,
      varlist=[var],
      text=(0.0,1.0,'MC-Bkg'),
      text_style=dict(ha='left',va='bottom'),
      figax=(fig,axs.flat[-1]),
      **kwargs,
    )
  def compare_samples(var, bkg=use_bkg, figax=None, efficiency=True, **kwargs):
    if figax is None:
      figax = study.get_figax(len(modules), dim=(-1,len(modules)))
    fig, axs = figax

    samples = signals

    if bkg:
      samples = [ sample+bkg for sample, bkg in zip(samples, bkgs)]
    label=list(modules.keys())

    for i, sample in enumerate(samples):
      ax = axs.flat[i] if len(modules) > 1 else axs
      study.quick(
        sample, legend=True, stacked=True,
        varlist=[var],
        # h_linestyle=[h_linestyle[i]]*sample.is_signal.npy.sum(),
        text=(0.0,1.0, label[i]),
        text_style=dict(ha='left',va='bottom'),
        efficiency=efficiency,
        figax=(fig,axs),
        **kwargs,
      )
  return compare_modules, compare_samples