import numpy as np
from pickle import load
from sklearn.preprocessing import MinMaxScaler

f = np.load('inputs/reco/nn_input_MX700_MY400_classifier.npz')

print(f.files)

X_train = np.column_stack((f['X_train'], f['m_train']))
X_test = np.column_stack((f['X_test'], f['m_test']))
X_val = np.column_stack((f['X_val'], f['m_val']))

# scaler = load(open('inputs/reco/nn_input_MX700_MY400_classifier_scaler.pkl', 'rb'))

scaler = MinMaxScaler()
scaler.fit(np.concatenate((f['m_train'], f['m_test'], f['m_val'])).reshape(-1,1))

m_train = scaler.transform(f['m_train'].reshape(-1,1))
m_test = scaler.transform(f['m_test'].reshape(-1,1))
m_val = scaler.transform(f['m_val'].reshape(-1,1))

M_train = f['m_train']
M_test  = f['m_test']
M_val   = f['m_val']

x_train = np.column_stack((f['x_train'], m_train))
x_test = np.column_stack((f['x_test'], m_test))
x_val = np.column_stack((f['x_val'], m_val))

np.savez('inputs/reco/nn_input_MX700_MY400_classifier_minv.npz', nonHiggs_indices=f['nonHiggs_indices'], X_train=X_train, X_test=X_test, X_val=X_val, x_train=x_train, x_test=x_test, x_val=x_val, y_train=f['y_train'], y_test=f['y_test'], y_val=f['y_val'], m_train=m_train, m_test=m_test, m_val=m_val, M_train=M_train, M_test=M_test, M_val=M_val, train=f['train'], val=f['val'], test=f['test'], train_pair_label=f['train_pair_label'], val_pair_label=f['val_pair_label'], test_pair_label=f['test_pair_label'], train_label=f['train_label'], val_label=f['val_label'], test_label=f['test_label'])