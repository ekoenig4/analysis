import sys
import git

sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils.eightbUtils.reco_genobjs import mass_list
import utils.torchUtils as gnn

from argparse import ArgumentParser 

# --- ARGS --- #
parser = ArgumentParser()
parser.add_argument('--epochs','-e',help='maximum number of epochs to run',type=int,default=10)
parser.add_argument('--nn1_out',help='GoldenClassifier.nn1_out',type=int,default=32)
parser.add_argument('--nn2_out',help='GoldenClassifier.nn2_out',type=int,default=128)

parser.add_argument('--train_size',help='Number of graphs to train with',type=int,default=-1)
parser.add_argument('--valid_size',help='Fraction of training graphs to use as validation',type=float,default=0.2)
parser.add_argument('--test_size',help='Fraction of testing graphs to use to test with',type=float,default=0.8)

parser.add_argument('--out','-o',help='Output directory for model',type=str,default='models/golden_classifier')

args = parser.parse_args()

# --- Loading Data --- #

print('Loading Training and Testing Data...')
template = gnn.Dataset('data/template',make_template=True, transform=gnn.Transform(gnn.to_uptri_graph()))

dataset = gnn.concat_dataset([f'data/{mass}-training' for mass in mass_list],transform=template.transform)

testing = gnn.concat_dataset([f'data/{mass}-testing' for mass in mass_list],transform=template.transform)

from torch_geometric.loader import DataLoader

training,validation = gnn.train_test_split(dataset,args.valid_size)

trainloader = DataLoader(training,batch_size=100,shuffle=True,num_workers=16)
validloader = DataLoader(validation,batch_size=100,num_workers=16)

testsample,_ = gnn.train_test_split(testing,1-args.test_size)
testloader = DataLoader(testsample,batch_size=100,num_workers=16)

# --- Loading Model --- #

print('Loading GoldenGCN Model...')
model = gnn.GoldenGCN(template,nn1_out=args.nn1_out,nn2_out=args.nn2_out,loss=gnn.losses.std_loss)
# model = gnn.GoldenGCN.load_from_checkpoint("models/golden_classifier/lightning_logs/version_3/checkpoints/epoch=9-step=39389.ckpt",dataset=template)

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

trainer = pl.Trainer(max_epochs=args.epochs, gpus=1 if gnn.useGPU else 0,
                     default_root_dir=args.out)

# --- Fitting Model --- #

fit = trainer.fit(model, trainloader, validloader)

test_results = trainer.test(model,testloader)
