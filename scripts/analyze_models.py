import os

from numpy.core.numerictypes import english_capitalize
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF CPU warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.rcParams['figure.autolayout'] = False
from argparse import ArgumentParser
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.models import model_from_json
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
# import matplotlib.cm as cm
import matplotlib.gridspec as gridspec

# custom libraries and modules
from logger import info, error
from consistent_plots import hist, hist2d
from kinematics import calcDeltaR
from colors import W, PURPLE


### ------------------------------------------------------------------------------------
## Implement command line parser

info("Parsing command line arguments.")

parser = ArgumentParser(description='Command line parser of model options and tags')
parser.add_argument('--type'   , dest = 'type'   , help = 'reco, parton, smeared'   ,  default = 'reco')
parser.add_argument('--task'   , dest = 'task'   , help = 'classifier or regressor' ,  default = 'classifier')
parser.add_argument('--nmodels', dest = 'nmodels', help = 'number of models trained',  default = 1    , type = int)
parser.add_argument('--tag'   , dest = 'tag'   , help = 'special tag'   ,  default = None)

args = parser.parse_args()

in_dir = f'models/{args.task}/{args.type}/'
if args.tag:
    in_dir += f'{args.tag}/'
model_dir = in_dir + 'model/'
eval_dir = f'plots/{args.task}/{args.type}/'

info(f"Evaluating {args.nmodels} models from location \n\t{PURPLE}{in_dir}{W}")

threshold = 0.5

print()
print('-'*80)
print(' '*25 + 'Neural Network Hyperparameters')
print('-'*80)

nn_info =  in_dir + 'nn_info.txt'
with open(nn_info, 'r') as nn_info:
    lines = nn_info.readlines()

for line in lines:
    print(line)
print('-'*80)

def get_scores(x_test, nmodel):

    nmodel = str(nmodel)

    info(f"Evaluating model {model_dir}model_{nmodel}.json")

    # load json and create model
    model_json_file = open(model_dir + f'model_{nmodel}.json', 'r')
    model_json = model_json_file.read()
    model_json_file.close()
    model = model_from_json(model_json)

    # load weights into new model
    model.load_weights(model_dir + f'model_{nmodel}.h5')

    scores = model.predict(x_test)

    return scores

def get_history(i):
    history_json_file = open(model_dir + 'history_' + str(i) + '.json', 'r')
    history = pd.read_json(history_json_file)
    history_json_file.close()
    return history


def get_roc(y, scores):
    fpr, tpr, thresholds_keras = roc_curve(y, scores)
    nn_auc = auc(fpr, tpr)
    return fpr, tpr, nn_auc


print()
info("Beginning model analysis.")

### ------------------------------------------------------------------------------------
## Load or generate scores npz file

# if not os.exists('Model_Eval'):

pdf_file = f'plots/{args.task}/{args.type}.pdf'
if args.tag:
    pdf_file = f'plots/{args.task}/{args.type}_{args.tag}.pdf'

with PdfPages(pdf_file) as pdf:

    for nmodel in np.arange(1,args.nmodels+1):

        ex_file = f"inputs/{args.type}/nn_input_MX700_MY400_{args.task}.npz"
        if args.tag:
            ex_file = f"inputs/{args.type}/nn_input_MX700_MY400_{args.task}_{args.tag}.npz"
        if args.tag == 'minv':
            ex_file = f"inputs/{args.type}/nn_input_MX700_MY400_{args.task}_minv.npz"
        info(f"Importing test set from file: \n\t{PURPLE}{ex_file}{W}")
        examples = np.load(ex_file)

        x = examples['x_test']
        y = examples['y_test']

        scores = get_scores(x, nmodel)

        np.savez(in_dir + 'scores_{nmodel}', scores=scores)

        d = {'Target':y[:,0],
             'Scores':np.around(scores[:,0], decimals=4),
             'Difference':np.around((y-scores)[:,0], decimals=4)}

        df =  pd.DataFrame(data=d)

        # df.style.background_gradient(cmap=cm.get_cmap('PiYG'), subset='Difference').render()

        print(df.head())

        if args.task == 'classifier':
            y_nH = y[:,0]
            y_H = y[:,0]
        else:
            y_nH = y
            y_H = y

        H_mask = (y_H == 1)
        nH_mask = (y_nH == 0)

        m = examples['m_test']
        if args.tag:
            m = examples['M_test']

        eta1 = examples['X_test'][:,1]
        phi1 = examples['X_test'][:,2]
        eta2 = examples['X_test'][:,4]
        phi2 = examples['X_test'][:,5]

        dR = calcDeltaR(eta1, eta2, phi1, phi2)
        print(dR.shape)
        print(scores.shape)

        dR_nH = dR[nH_mask]
        dR_H = dR[H_mask]

        if args.task == 'classifier':
            p_nH = scores[:,0][nH_mask]
            p_H = scores[:,0][H_mask]

            m_nH = m[nH_mask]
            m_H = m[H_mask]

            high_nH = p_nH > threshold
            low_H   = p_H  < threshold
        else:
            p_nH = scores[nH_mask]
            p_H = scores[H_mask]

        


        ### ------------------------------------------------------------------------------------
        ## Prep plots

        dR_bins = np.linspace(0, 5.5, 100)
        m_bins = np.linspace(0, 500, 100)
        p_bins = np.linspace(0,1,100)

        fig = plt.figure(figsize=(18,10))
        # fig.suptitle(f"{args.type}, {english_capitalize(args.task)}, Training Session #{nmodel}")

        gs = gridspec.GridSpec(nrows=3, ncols=4, height_ratios=[4,4,5])

        mnH_ax       = fig.add_subplot(gs[1,0])
        mH_ax        = fig.add_subplot(gs[1,1])
        dRnH_ax      = fig.add_subplot(gs[1,2])
        dRH_ax       = fig.add_subplot(gs[1,3])

        mscatnH_ax   = fig.add_subplot(gs[2, 0])
        mscatH_ax    = fig.add_subplot(gs[2, 1])
        dRscatnH_ax  = fig.add_subplot(gs[2, 2])
        dRscatH_ax   = fig.add_subplot(gs[2, 3])

        scorenH_ax1  = fig.add_subplot(gs[0, 0], label="Regular")
        # scorenH_ax2  = fig.add_subplot(gs[0, 0], label="LogPlot", frameon=False)
        scoreH_ax    = fig.add_subplot(gs[0, 1], label="Higgs")

        roc_ax   = fig.add_subplot(gs[0, 2])
        history_ax = fig.add_subplot(gs[0, 3])

        # line = plt.Line2D((.48,.48),(.1,.92), color="k", linewidth=1)
        # fig.add_artist(line)

        # line = plt.Line2D((.05,.95),(.92,.92), color="k", linewidth=1)
        # fig.add_artist(line)

        ### ------------------------------------------------------------------------------------
        ## Plot distribution of test masses

        # hist(mH_ax, m_H[low_H], bins=m_bins, label='Low Scores')
        # hist(mnH_ax, m_nH[high_nH], bins=m_bins, label='Low Scores')

        hist(mH_ax, m_H, bins=m_bins, label='All Scores')
        hist(mnH_ax, m_nH, bins=m_bins, label='All Scores')

        mH_ax.text(0.1, 1.02, 'Higgs Pair', fontsize='small', transform=mH_ax.transAxes)
        mnH_ax.text(0.1, 1.02, 'Non-Higgs Pair', fontsize='small', transform=mnH_ax.transAxes)

        mH_ax.set_xlabel(r'Higgs Pair $m_{bb}$ [GeV]')
        mH_ax.set_ylabel('Count')
        
        mH_ax.yaxis.get_offset_text().set_visible(False)
        mnH_ax.yaxis.get_offset_text().set_visible(False)

        mH_ax.text(-0.05, 1.02, r'$\times10^{3}$', fontsize='smaller', transform=mH_ax.transAxes)
        mnH_ax.text(-0.05, 1.02, r'$\times10^{3}$', fontsize='smaller', transform=mnH_ax.transAxes)

        mnH_ax.set_xlabel(r'Non-Higgs Pair $m_{bb}$ [GeV]')
        mnH_ax.set_ylabel('Count')

        # mnH_ax.legend()

        ### ------------------------------------------------------------------------------------
        ## Plot distribution of test DeltaR

        dRH_ax.text(0.1, 1.02, 'Higgs Pair', fontsize='small', transform=dRH_ax.transAxes)
        dRnH_ax.text(0.1, 1.02, 'Non-Higgs Pair', fontsize='small', transform=dRnH_ax.transAxes)

        hist(dRH_ax, dR_H, bins=dR_bins, label='All Scores')
        # hist(dRH_ax, dR_H[low_H], bins=dR_bins, label='Low Scores')
        hist(dRnH_ax, dR_nH, bins=dR_bins, label='All Scores')
        # hist(dRnH_ax, dR_nH[high_nH], bins=dR_bins, label='Low Scores')

        dRH_ax.yaxis.get_offset_text().set_visible(False)
        dRH_ax.text(-0.05, 1.02, r'$\times10^{3}$', fontsize='smaller', transform=dRH_ax.transAxes)

        # dRnH_ax.legend()
        

        dRH_ax.set_xlabel(r"Higgs pair $\Delta R_{bb}$")
        dRH_ax.set_ylabel('Count')

        dRnH_ax.set_xlabel(r"Non-Higgs pair $\Delta R_{bb}$")
        dRnH_ax.set_ylabel('Count')


        ### ------------------------------------------------------------------------------------
        ## Sample mass scatter plot for the same random model

        mscatH_ax.text(0, 1.02, 'Higgs Pair', fontsize='small', transform=mscatH_ax.transAxes)
        mscatnH_ax.text(0, 1.02, 'Non-Higgs Pair', fontsize='small', transform=mscatnH_ax.transAxes)

        n, xedges, yedges, im = hist2d(mscatnH_ax, m_nH, p_nH, xbins=m_bins, ybins=p_bins)
        mscatnH_ax.set_xlabel(r"Non-Higgs pair $m_{bb}$ [GeV]")
        mscatnH_ax.set_ylabel("NN Score")
        fig.colorbar(im, ax=mscatnH_ax)

        n, xedges, yedges, im = hist2d(mscatH_ax, m_H, p_H, xbins=m_bins, ybins=p_bins)
        mscatH_ax.set_xlabel(r"Higgs pair $m_{bb}$ [GeV]")
        mscatH_ax.set_ylabel("NN Score")
        fig.colorbar(im, ax=mscatH_ax)


        ### ------------------------------------------------------------------------------------
        ## Sample DeltaR scatter plot for the same random model

        dRscatH_ax.text(0, 1.02, 'Higgs Pair', fontsize='small', transform=dRscatH_ax.transAxes)
        dRscatnH_ax.text(0, 1.02, 'Non-Higgs Pair', fontsize='small', transform=dRscatnH_ax.transAxes)

        n, xedges, yedges, im = hist2d(dRscatnH_ax, dR_nH, p_nH, xbins=dR_bins, ybins=p_bins)
        dRscatnH_ax.set_xlabel(r"Non-Higgs pair $\Delta R_{bb}$")
        dRscatnH_ax.set_ylabel("NN Score")
        fig.colorbar(im, ax=dRscatnH_ax)

        n, xedges, yedges, im = hist2d(dRscatH_ax, dR_H, p_H, xbins=dR_bins, ybins=p_bins)
        dRscatH_ax.set_xlabel(r"Higgs pair $\Delta R_{bb}$")
        dRscatH_ax.set_ylabel("NN Score")
        fig.colorbar(im, ax=dRscatH_ax)


        ### ------------------------------------------------------------------------------------
        ## Plot distribution of NN Test Scores


        hist(scoreH_ax, p_H, bins=p_bins, label='True Higgs Pair')
        hist(scorenH_ax1, p_nH, bins=p_bins, label='True Non-Higgs Pair')
        # hist(scorenH_ax2, p_nH, bins=p_bins, label='True Non-Higgs Pair', color=c2)

        scoreH_ax.text(0.1, 1.02, 'Higgs Pair', fontsize='small', transform=scoreH_ax.transAxes)
        scorenH_ax1.text(0.1, 1.02, 'Non-Higgs Pair', fontsize='small', transform=scorenH_ax1.transAxes)

        scoreH_ax.set_xlabel('NN Score')
        scoreH_ax.set_ylabel('Count')

        scorenH_ax1.set_xlabel('NN Score')
        scorenH_ax1.set_ylabel('Count')
        # scorenH_ax1.tick_params(axis='y', colors=c1)

        # scorenH_ax2.tick_params(axis='x', labelbottom=False)
        # scorenH_ax2.set_yscale('log')
        # # scorenH_ax2.set_ylabel('Log(Count)', rotation=270, labelpad=25)
        # scorenH_ax2.yaxis.tick_right()
        # scorenH_ax2.yaxis.set_label_position('right')
        # scorenH_ax2.tick_params(axis='y',  colors=c2)
        # scorenH_ax2.axis()

        scoreH_ax.yaxis.get_offset_text().set_visible(False)
        scorenH_ax1.yaxis.get_offset_text().set_visible(False)

        scoreH_ax.text(-0.05, 1.02, r'$\times10^{3}$', fontsize='smaller', transform=scoreH_ax.transAxes)
        scorenH_ax1.text(-0.05, 1.02, r'$\times10^{4}$', fontsize='smaller', transform=scorenH_ax1.transAxes, color='C0')

        ### ------------------------------------------------------------------------------------
        ## Sample the ROC curve for a random model

        fpr, tpr, nn_auc = get_roc(y[:,0], scores[:,0])

        roc_ax.plot(fpr, tpr, label=f'auc = {nn_auc:.3f})')
        roc_ax.set_xlabel('False positive rate')
        roc_ax.set_ylabel('True positive rate')
        roc_ax.legend()

        ### ------------------------------------------------------------------------------------
        ## Sample the ROC curve for a random model

        history = get_history(nmodel)

        train_acc = history['loss']
        valid_acc = history['val_loss']
        nepochs = np.arange(len(train_acc))

        history_ax.set_ylim(0,1.05)
        history_ax.plot(nepochs, train_acc, label='Training')
        history_ax.plot(nepochs, valid_acc, label='Validation')
        history_ax.set_xlabel('Epoch')
        history_ax.set_ylabel('Loss')
        history_ax.legend()

        plt.tight_layout()


        ### ------------------------------------------------------------------------------------
        ## Save!
        info(f"Saving pdf to \n\t{PURPLE}{pdf_file}{W}")
        pdf.savefig()