import numpy as np

def get_abcd_masks(v1_r, v2_r, v1='n_medium_btag', v2='quadh_score'):
  v1_sr = lambda t : (t[v1] >= v1_r[1]) & (t[v1] < v1_r[2])
  v1_cr = lambda t : (t[v1] >= v1_r[0]) & (t[v1] < v1_r[1])

  v2_sr = lambda t : (t[v2] >= v2_r[1]) & (t[v2] < v2_r[2])
  v2_cr = lambda t : (t[v2] >= v2_r[0]) & (t[v2] < v2_r[1])

  r_a = lambda t : v1_sr(t) & v2_sr(t)
  r_b = lambda t : v1_cr(t) & v2_sr(t)

  r_c = lambda t : v1_sr(t) & v2_cr(t)
  r_d = lambda t : v1_cr(t) & v2_cr(t)
  return r_a, r_b, r_c, r_d

def get_region_scale(r, model):
  t = model.apply(lambda t:t.scale).apply(np.sum).npy.sum()
  n = model.apply(lambda t:t.scale[r(t)]).apply(np.sum).npy.sum()
  e = np.sqrt(model.apply(lambda t:(t.scale[r(t)])**2).apply(np.sum).npy.sum())
  return n/t,e/t


def get_abcd_scale(r_a, r_b, r_c, r_d, model):
  n_d, e_d = get_region_scale(r_d, model)
  n_c, e_c = get_region_scale(r_c, model)
  n_b, e_b = get_region_scale(r_b, model)
  n_a, e_a = get_region_scale(r_a, model)

  # print(n_a, n_b, n_c, n_d)
  k_factor = n_c/n_d
  e_factor = k_factor*np.sqrt( (e_c/n_c)**2 + (e_d/n_d)**2 )
  k_target = n_a/n_b
  e_target = k_target*np.sqrt( (e_a/n_a)**2 + (e_b/n_b)**2 )

  n_model = k_factor*n_b
  e_model = n_model*np.sqrt( (e_factor/k_factor)**2 )

  return (k_target, e_target), (k_factor, e_factor), (n_model, e_model)