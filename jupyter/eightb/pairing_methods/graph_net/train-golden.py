import sys
import git

sys.path.append( git.Repo('.', search_parent_directories=True).working_tree_dir )
from utils.eightbUtils.reco_genobjs import mass_list
import utils.torchUtils as gnn

import warnings
warnings.filterwarnings("ignore") 

import multiprocessing
ncpu = multiprocessing.cpu_count()

import torch
torch.multiprocessing.set_sharing_strategy('file_system')


from argparse import ArgumentParser 

# --- ARGS --- #
parser = ArgumentParser()

parser.add_argument('--model',help='Select model to train',type=str,required=True,choices=gnn.modelMap.keys())
parser.add_argument('--model-args',help='Specify model args',type=int,nargs='+',default=[])
parser.add_argument('--output',help='Specify output directory for lightning logs',type=str, default='models')

parser.add_argument('--epochs',help='Maximum number of epochs to run',type=int,default=100)
parser.add_argument('--loss',help='Specify loss function to train with',type=str,default='std_loss')
parser.add_argument('--lr', help='Specify learning rate', type=float, default=1e-3)
parser.add_argument('--scale',help='Specify how to scale features',type=str,default='standardize',choices=['raw','normalize','standardize'])
parser.add_argument('--node-mask',help='Specify what node features to use',nargs='+',type=str,default=[])
parser.add_argument('--edge-mask',help='Specify what edge features to use',nargs='+',type=str,default=[])

parser.add_argument('--uptri',help='Use upper triangular adjacency matrix',default=False,action='store_true')
parser.add_argument('--cluster-y',help='Calculate Cluster Y',default=False,action='store_true')
parser.add_argument('--hyper-edge',help='Construct all possible 4 node hyper edges',default=False,action='store_true')
parser.add_argument('--min-knn',help='Construct minimum knn graph',type=int, default=0)
parser.add_argument('--remove-self',help='Remove self loop from convolutions',default=False,action='store_true')

# parser.add_argument('--train-size',help='Number of graphs to train with',type=int,default=-1) #! not implemented
parser.add_argument('--valid-size',help='Fraction of training graphs to use as validation',type=float,default=0.2)
parser.add_argument('--test-size',help='Fraction of testing graphs to use to test with',type=float,default=0.8)
parser.add_argument('--batch-size',help='Specify batch size',type=int, default=1000)

parser.add_argument('--no-gpu',help='Dont use GPU',default=True,action='store_false')
args = parser.parse_args()

gnn.config.set_gpu(args.no_gpu)

# --- Loading Data --- #

print('Loading Training and Testing Data...')
print('N CPU:',ncpu)

hparams = dict()
transform = gnn.Transform()
if args.uptri: transform.append(gnn.to_uptri_graph())
if args.cluster_y: transform.append(gnn.cluster_y())
if args.hyper_edge: transform.append(gnn.HyperEdgeY())
if args.remove_self: transform.append(gnn.remove_self_loops())
if args.min_knn > 0: 
    hparam = dict(n_neighbor=args.min_knn)
    transform.append(gnn.min_edge_neighbor(**hparam))
    hparams.update(hparam)

template = gnn.Dataset('data/template',make_template=True, transform=transform, scale=args.scale, node_mask=args.node_mask, edge_mask=args.edge_mask)

dataset = gnn.concat_dataset([f'data/{mass}-training' for mass in mass_list],transform=template.transform)

testing = gnn.concat_dataset([f'data/{mass}-testing' for mass in mass_list],transform=template.transform)

from torch_geometric.loader import DataLoader

training,validation = gnn.train_test_split(dataset,args.valid_size)

trainloader = DataLoader(training,batch_size=args.batch_size,shuffle=True,num_workers=ncpu)
validloader = DataLoader(validation,batch_size=args.batch_size,num_workers=ncpu)

testsample,_ = gnn.train_test_split(testing,1-args.test_size)
testloader = DataLoader(testsample,batch_size=args.batch_size,num_workers=ncpu)

# --- Loading Model --- #

print('Loading GCN Model...')

model = gnn.modelMap[args.model](*args.model_args,dataset=template,loss=args.loss,lr=args.lr,batch_size=args.batch_size, hparams=hparams)

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

trainer = pl.Trainer(
    max_epochs=args.epochs,
    gpus=-1 if gnn.config.useGPU else 0,
    accelerator="auto",
    # precision=16,
    callbacks=[EarlyStopping(
        monitor="hp/loss",
        stopping_threshold=1e-4,
        divergence_threshold=6.0)],
    default_root_dir=f'{args.output}/{model.name}'
)

# --- Fitting Model --- #

fit = trainer.fit(model, trainloader, validloader)

test_results = trainer.test(model,testloader)
