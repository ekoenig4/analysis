# This script is used to construct the inputs used to train the neural network (NN).

from modules.colors import CYAN, W
from modules.kinematics import calcDeltaR
from modules.logger import info
from modules.myuproot import open_up
 
import numpy as np
import os
import uproot3_methods
import vector
from argparse import ArgumentParser
from keras.models import model_from_json
from pickle import dump
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time

### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')

parser.add_argument('--type'      , dest = 'type'   , help = 'parton or reco'       , default = 'reco' )
parser.add_argument('--task'      , dest = 'task'   , help = 'class or reg?'        , default = 'classifier' )
parser.add_argument('--MX'        , dest = 'MX'     , help = 'mass of X resonance'  , default = 700   )
parser.add_argument('--MY'        , dest = 'MY'     , help = 'mass of Y resonance'  , default = 400   )
parser.add_argument('--no_presel' , dest = 'presel' , help = 'apply preselections?' , default = True  , action = 'store_false')
parser.add_argument('--pair'      , dest = 'pair'   , help = 'specify which pair to find (HX,HY1,HY2), default all', choices=['HX','HY1','HY2',"ALL"], default = "ALL")
parser.add_argument('--no-save'   , dest = 'save'   , help = 'disable saving'       , default = True , action = 'store_false')

args = parser.parse_args()


### ------------------------------------------------------------------------------------
## Load signal events

MX = args.MX
MY = args.MY

cwd = os.getcwd()
dir_prefix = 'inputs/'

reco_filename = f'signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_reco_preselections_500k.root'
info(f"Opening ROOT file {CYAN}{reco_filename}{W} with columns")
table =  open_up(reco_filename, 'sixBtree')
nevents = table._length()

### ------------------------------------------------------------------------------------
## Prepare bs for pairing

pairmap = {
    "HX": [1,0,0],
    "HY1":[0,1,0],
    "HY2":[0,0,1],
    "ALL":[1,1,1]
}
pair_tag = args.pair
output_signal_mask = pairmap[pair_tag]

if args.type == 'parton':
    tag1 = ''
    tag2 = ''
if args.type == 'reco':
    tag1 = '_recojet'
    tag2 = 'Regressed'

HX_b1  = {'pt': table[f'HX_b1{tag1}_pt{tag2}' ],
          'eta':table[f'HX_b1{tag1}_eta'],
          'phi':table[f'HX_b1{tag1}_phi'],
          'm':  table[f'HX_b1{tag1}_m'  ]}
HX_b2  = {'pt': table[f'HX_b2{tag1}_pt{tag2}' ],
          'eta':table[f'HX_b2{tag1}_eta'],
          'phi':table[f'HX_b2{tag1}_phi'],
          'm':  table[f'HX_b2{tag1}_m'  ]}
HY1_b1 = {'pt': table[f'HY1_b1{tag1}_pt{tag2}'],
          'eta':table[f'HY1_b1{tag1}_eta'],
          'phi':table[f'HY1_b1{tag1}_phi'],
          'm':  table[f'HY1_b1{tag1}_m' ]}
HY1_b2 = {'pt': table[f'HY1_b2{tag1}_pt{tag2}'],
          'eta':table[f'HY1_b2{tag1}_eta'],
          'phi':table[f'HY1_b2{tag1}_phi'],
          'm':  table[f'HY1_b2{tag1}_m' ]}
HY2_b1 = {'pt': table[f'HY2_b1{tag1}_pt{tag2}'],
          'eta':table[f'HY2_b1{tag1}_eta'],
          'phi':table[f'HY2_b1{tag1}_phi'],
          'm':  table[f'HY2_b1{tag1}_m' ]}
HY2_b2 = {'pt': table[f'HY2_b2{tag1}_pt{tag2}'],
          'eta':table[f'HY2_b2{tag1}_eta'],
          'phi':table[f'HY2_b2{tag1}_phi'],
          'm':  table[f'HY2_b2{tag1}_m' ]}

part_dict = {0:HX_b1, 1:HX_b2, 2:HY1_b1, 3:HY1_b2, 4:HY2_b1, 5:HY2_b2}
part_name = {0:'HX_b1', 1:'HX_b2', 2:'HY1_b1', 3:'HY1_b2', 4:'HY2_b1', 5:'HY2_b2'}
pair_dict = {0:1, 1:0, 2:3, 3:2, 4:5, 5:4} # Used later to verify that non-Higgs
                                           # pair candidates are truly non-Higgs pairs

nonHiggs_labels = np.array((
 'X b1, Y1 b1',
 'X b1, Y1 b2',
 'X b1, Y2 b1',
 'X b1, Y2 b2',
 'X b2, Y1 b1',
 'X b2, Y1 b2',
 'X b2, Y2 b1',
 'X b2, Y2 b2',
 'Y1 b1, Y2 b1',
 'Y1 b1, Y2 b2',
 'Y1 b2, Y2 b1',
 'Y1 b2, Y2 b2',))

info(f"Files contain {nevents} events.")


### ------------------------------------------------------------------------------------
## Classifier

if args.task == 'classifier':

    evt_indices = np.arange(nevents)
    test_size = 0.20
    val_size = 0.125
    evt_train, evt_test = train_test_split(evt_indices, test_size=test_size)
    evt_train, evt_val = train_test_split(evt_train, test_size=val_size)

    ntrain = len(evt_train)
    ntest  = len(evt_test)
    nval   = len(evt_val)

    info(f"Building Classifier for {pair_tag} pairs")
    info(f"Number of examples in training set:   {ntrain}")
    info(f"Number of examples in testing set:    {ntest}")
    info(f"Number of examples in validation set: {nval}")

    # mask for non-Higgs pairs, contains 3 True and 9 False elements
    nonHiggs_mask = [True]*3 + [False]*9

    def randomizer():
        indices = np.array((), dtype=int)
        mask = np.array((), dtype=bool)
        for i in np.arange(nevents):
            # shuffle nonHiggs_mask to select three at random
            shuffle(nonHiggs_mask)
            mask = np.append(mask, nonHiggs_mask)
            indices = np.append(indices, [i for i,b in enumerate(nonHiggs_mask) if b == True])
        indices = indices.reshape(nevents, 3)
        mask = mask.reshape(nevents, 12)
        return mask, indices

    random_mask, random_indices = randomizer()

    info("Generating feature block.")

    # Define input variables for pair of bjets 
    def getInputs(b1_dict,b2_dict):
        b1 = vector.obj(pt=b1_dict['pt'], 
                        eta=b1_dict['eta'], 
                        phi=b1_dict['phi'], 
                        mass=np.repeat(4e-9, nevents))
        b2 = vector.obj(pt=b2_dict['pt'], 
                        eta=b2_dict['eta'], 
                        phi=b2_dict['phi'],  
                        mass=np.repeat(4e-9, nevents))

        H_candidate = b1 + b2
        invm = H_candidate.mass
        
        boosted_b1 = b1.boost(-H_candidate)
        boosted_b2 = b2.boost(-H_candidate)
        
        dR = b1.deltaR(b2)
        return np.column_stack((b1.pt, boosted_b1.pt, b1.eta, b1.phi, b2.pt, boosted_b2.pt, b2.eta, b2.phi, H_candidate.pt, invm, dR))
    # Input variable key list to help identify input variables later
    input_keys =          ["b1_pt","boosted_b1_pt","b1_eta","b1_phi","b2_pt","boosted_b2_pt","b2_eta","b2_phi","higgs_pt","higgs_mass","deltaR"]
    
    X_truH = [] # List for true higgs pairs
    X_nonH = [] # List for non higgs pairs
    for i in range(5):
        b1_dict = part_dict[i]
        for j in range(i+1,6):
            b2_dict = part_dict[j]
            pair_name = part_name[i] + " & " + part_name[j]
            # print(f"Generating features for pair: {pair_name}")
            inputs = getInputs(b1_dict,b2_dict)

            # tru higgs pairs: 0-1, 2-3, 4-5 
            if j-i == 1 and i % 2 == 0:
                  X_truH.append(inputs)
            else: X_nonH.append(inputs)

    X_train = []
    X_val = []
    X_test = []
    
    y_train = np.array(())
    y_val = np.array(())
    y_test = np.array(())

    train_pair_label = np.array(())
    val_pair_label = np.array(())
    test_pair_label = np.array(())

    train_label = np.array(())
    val_label = np.array(())
    test_label = np.array(())

    info("Generating training examples.")
    start = time.perf_counter()

    output_list = output_signal_mask + [0]*3
    for i in tqdm(evt_train):
        input_list = [ truH[i,:] for truH in X_truH ] + [ X_nonH[rj][i,:] for rj in random_indices[i] ]
        
        X_train += input_list
        y_train = np.append(y_train, np.array(output_list))
        train_pair_label = np.append(train_pair_label, np.concatenate((np.array(('HX', 'HY1', 'HY2')), nonHiggs_labels[random_indices[i,:]])))
        train_label = np.append(train_label, np.array((['Higgs']*3, ['Non-Higgs']*3)))
    X_train = np.vstack(X_train)    
    y_train = np.vstack((y_train, np.where(y_train == 0, 1, 0))).T

    info(f"Total Elapsed Time: {time.perf_counter()-start:0.4f}s")
    info(f"--> X Train Shape: {X_train.shape}\n    Y Train Shape: {y_train.shape}")
    
    info("Generating validation features.")
    start = time.perf_counter()
    
    for i in tqdm(evt_val):
        input_list = [ truH[i,:] for truH in X_truH ] + [ X_nonH[rj][i,:] for rj in random_indices[i] ]
        
        X_val += input_list
        y_val = np.append(y_val, np.array(output_list))
        nH_label = nonHiggs_labels[random_indices[i,:]]
        val_pair_label = np.append(val_pair_label, np.concatenate((np.array(('HX', 'HY1', 'HY2')), nH_label)))
        val_label = np.append(val_label, np.array((['Higgs']*3, ['Non-Higgs']*3)))
    X_val = np.vstack(X_val)
    y_val = np.vstack((y_val, np.where(y_val == 0, 1, 0))).T
    
    info(f"Total Elapsed Time: {time.perf_counter()-start:0.4f}s")
    info(f"--> X Val Shape: {X_val.shape}\n    Y Val Shape: {y_val.shape}")

    info("Generating testing features.")
    start = time.perf_counter()

    output_list = output_signal_mask + [0]*12
    for i in tqdm(evt_test):
        input_list = [ truH[i,:] for truH in X_truH ]+[ nonH[i,:] for nonH in X_nonH ]

        X_test += input_list
        y_test = np.append(y_test, np.array(output_list))
        test_pair_label = np.append(test_pair_label, np.concatenate((np.array(('HX', 'HY1', 'HY2')), nonHiggs_labels)))
        labels = ['Higgs']*3 + ['Non-Higgs']*12
        test_label = np.append(test_label, np.array((labels)))
    X_test = np.vstack(X_test)
    y_test = np.vstack((y_test, np.where(y_test == 0, 1, 0))).T
    
    info(f"Total Elapsed Time: {time.perf_counter()-start:0.4f}s")
    info(f"--> X Test Shape: {X_test.shape}\n    Y Test Shape: {y_test.shape}")

    assert X_train.shape[0] == y_train.shape[0], print(X_train.shape[0], y_train.shape[0])
    assert X_val.shape[0] == y_val.shape[0], print(X_val.shape[0], y_val.shape[0])
    assert X_test.shape[0] == y_test.shape[0], print(X_test.shape[0], y_test.shape[0])

    info("Normalizing the examples.")
    scaler = MinMaxScaler()
    scaler.fit(np.vstack((X_train, X_val, X_test)))

    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    x_val = scaler.transform(X_val)

    if args.save:
        filename = dir_prefix + f"{args.type}/nn_input_MX{args.MX}_MY{args.MY}_classifier_{pair_tag}"
        info(f"Saving training examples to {filename}.npz")
        
        scaler_file = dir_prefix + f'{args.type}/nn_input_MX{args.MX}_MY{args.MY}_classifier_{pair_tag}_scaler.pkl'
        info(f"Saving training example scaler to {scaler_file}")
        dump(scaler, open(scaler_file, 'wb'))
        
        np.savez(filename, nonHiggs_indices=random_indices,
                 X_train=X_train,  X_test=X_test, X_val=X_val, 
                 x_train=x_train,  x_test=x_test, x_val=x_val,  
                 y_train=y_train, y_test=y_test, y_val=y_val,
                 train=evt_train, val=evt_val, test=evt_test,
                 train_pair_label=train_pair_label, val_pair_label=val_pair_label, test_pair_label=test_pair_label,
                 train_label=train_label, val_label=val_label, test_label=test_label,input_keys=input_keys)

### ------------------------------------------------------------------------------------
## Regressor

elif args.task == 'regressor':

    ex_file = f"inputs/reco/nn_input_MX700_MY400_{args.task}.npz"
    info(f"Importing test set from file: {ex_file}")
    examples = np.load(ex_file)

    x = examples['x_test']
    y = examples['y_test']

    model_dir = f'models/classifier/{args.type}/model/'
    info(f"Evaluating model {model_dir}model_1.json")

    # load json and create model
    model_json_file = open(model_dir + f'model_1.json', 'r')
    model_json = model_json_file.read()
    model_json_file.close()
    model = model_from_json(model_json)

    # load weights into new model
    model.load_weights(model_dir + f'model_1.h5')

    evt_indices = np.arange(nevents)
    test_size = 0.20
    val_size = 0.125
    evt_train, evt_test = train_test_split(evt_indices, test_size=test_size)
    evt_train, evt_val = train_test_split(evt_train, test_size=val_size)

    ntrain = len(evt_train)
    ntest  = len(evt_test)
    nval   = len(evt_val)

    info(f"Number of examples in training set:   {ntrain}")
    info(f"Number of examples in testing set:    {ntest}")
    info(f"Number of examples in validation set: {nval}")

    class_scores = []
    Higgs_p4 = []
    for i in range(0,5,2):
        j = i+2
        b1 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(part_dict[i]['pt'], part_dict[i]['eta'], part_dict[i]['phi'], np.repeat(4e-9, nevents))
        b2 = uproot3_methods.TLorentzVectorArray.from_ptetaphim(part_dict[j]['pt'], part_dict[j]['eta'], part_dict[j]['phi'],  np.repeat(4e-9, nevents))
        dR = calcDeltaR(part_dict[i]['eta'], part_dict[j]['eta'], part_dict[i]['phi'], part_dict[j]['phi'])
        ins = np.column_stack((part_dict[i]['pt'], part_dict[i]['eta'], part_dict[i]['phi'], part_dict[j]['pt'], part_dict[j]['eta'], part_dict[j]['phi'], dR))
        scores = model.predict(ins)
        print(scores)
        scores = scores[:,0]
        class_scores.append(scores)

        H = b1 + b2
        Higgs_p4.append(H)

    inputs = []
    nonY_m = []
    for i in range(2):
        for j in range(i+1,3):
            H1 = Higgs_p4[i]
            H2 = Higgs_p4[j]
            H1_scores = class_scores[i]
            H2_scores = class_scores[j]

            HH = H1 + H2
            if i == 0: # i = 0 refers to HX_b1, which cannot make a 
                nonY_m.append(HH.mass)
            else:
                Y_m = HH.mass

            dR = calcDeltaR(H1.eta, H2.eta, H1.phi, H2.phi)

            inputs.append(np.column_stack((H1.pt, H1.eta, H1.phi, H2.pt, H2.eta, H2.phi, dR, H1_scores, H2_scores)))

    X_train = inputs[2][evt_train, :]
    X_val = inputs[2][evt_val,  :]
    X_test = np.vstack((inputs[2][evt_test , :], inputs[0], inputs[1]))

    m_test = np.concatenate((Y_m[evt_test], nonY_m[0], nonY_m[1]))

    scaler = MinMaxScaler()

    X  = np.vstack((X_train, X_val, X_test))

    scaler.fit(X)

    x_train = scaler.transform(X_train)
    x_val = scaler.transform(X_val)
    x_test = scaler.transform(X_test)

    y_train = np.repeat(400, len(X_train))
    y_val = np.repeat(400, len(X_val))
    y_test = np.concatenate((np.repeat(400, len(evt_test)), nonY_m[0], nonY_m[1]))

    assert(X_train.shape[0] == y_train.shape[0])
    assert(X_val.shape[0] == y_val.shape[0])
    assert(X_test.shape[0] == y_test.shape[0])

    if args.save:
        filename = dir_prefix + f"{args.type}/nn_input_MX{args.MX}_MY{args.MY}_regressor"
        info(f"Saving training examples to {filename}.npz")
        
        scaler_file = dir_prefix + f'{args.type}/nn_input_MX{args.MX}_MY{args.MY}_regressor_scaler.pkl'
        info(f"Saving training example scaler to {scaler_file}")
        dump(scaler, open(scaler_file, 'wb'))
        
        np.savez(filename, m_test = m_test,
                 X_train=X_train, x_train=x_train,
                 X_val=X_val, x_val=x_val,
                 X_test=X_test, x_test=x_test,
                 y_train=y_train, y_test=y_test, y_val=y_val,
                 train=evt_train, val=evt_val, test=evt_test)
