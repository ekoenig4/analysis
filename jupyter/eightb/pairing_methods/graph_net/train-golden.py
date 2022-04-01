import sys
import git

sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils.eightbUtils.reco_genobjs import mass_list
import utils.torchUtils as gnn

import warnings
warnings.filterwarnings("ignore") 

import multiprocessing
ncpu = multiprocessing.cpu_count()


from argparse import ArgumentParser 

# --- ARGS --- #
parser = ArgumentParser()
parser.add_argument('--epochs',help='Maximum number of epochs to run',type=int,default=100)
parser.add_argument('--nn1_out',help='GoldenClassifier.nn1_out',type=int,default=32)
parser.add_argument('--nn2_out',help='GoldenClassifier.nn2_out',type=int,default=128)
parser.add_argument('--loss',help='Specify loss function to train with',type=str,default='std_loss',choices=gnn.losses.lossMap.keys())
parser.add_argument('--lr', help='Specify learning rate', type=float, default=1e-3)
parser.add_argument('--scale',help='Specify how to scale features',type=str,default='standardize',choices=['raw','normalize','standardize'])
parser.add_argument('--node-mask',help='Specify what node features to use',nargs='+',type=str,default=[])
parser.add_argument('--edge-mask',help='Specify what edge features to use',nargs='+',type=str,default=[])

parser.add_argument('--uptri',help='Use upper triangular adjacency matrix',default=False,action='store_true')

# parser.add_argument('--train-size',help='Number of graphs to train with',type=int,default=-1) #! not implemented
parser.add_argument('--valid-size',help='Fraction of training graphs to use as validation',type=float,default=0.2)
parser.add_argument('--test-size',help='Fraction of testing graphs to use to test with',type=float,default=0.8)
parser.add_argument('--batch-size',help='Specify batch size',type=int, default=1000)

parser.add_argument('--model',help='Select model to train',type=str,default='golden',choices=gnn.modelMap.keys())

parser.add_argument('--no-gpu',help='Dont use GPU',default=True,action='store_false')
args = parser.parse_args()

gnn.config.set_gpu(args.no_gpu)

# --- Loading Data --- #

print('Loading Training and Testing Data...')
print('N CPU:',ncpu)

transform = None
if args.uptri: transform = gnn.to_uptri_graph()

template = gnn.Dataset('data/template',make_template=True, transform=transform, scale=args.scale, node_mask=args.node_mask, edge_mask=args.edge_mask)
print(template.transform.transforms)
exit()
dataset = gnn.concat_dataset([f'data/{mass}-training' for mass in mass_list],transform=template.transform)

testing = gnn.concat_dataset([f'data/{mass}-testing' for mass in mass_list],transform=template.transform)

from torch_geometric.loader import DataLoader

training,validation = gnn.train_test_split(dataset,args.valid_size)

trainloader = DataLoader(training,batch_size=args.batch_size,shuffle=True,num_workers=ncpu)
validloader = DataLoader(validation,batch_size=args.batch_size,num_workers=ncpu)

testsample,_ = gnn.train_test_split(testing,1-args.test_size)
testloader = DataLoader(testsample,batch_size=args.batch_size,num_workers=ncpu)

# --- Loading Model --- #

print('Loading GoldenGCN Model...')

model = gnn.modelMap[args.model](template,nn1_out=args.nn1_out,nn2_out=args.nn2_out,loss=args.loss,lr=args.lr,batch_size=args.batch_size)

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

trainer = pl.Trainer(
    max_epochs=args.epochs,
    gpus=-1 if gnn.config.useGPU else 0,
    accelerator="auto",
    precision=16,
    callbacks=[EarlyStopping(
        monitor="hp/loss",
        stopping_threshold=1e-4,
        divergence_threshold=6.0)],
    default_root_dir=model.output
)

# --- Fitting Model --- #

fit = trainer.fit(model, trainloader, validloader)

test_results = trainer.test(model,testloader)
