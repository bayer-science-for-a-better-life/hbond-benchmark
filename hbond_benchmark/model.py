from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
from pytorch_lightning import LightningModule, LightningDataModule
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
import torch
from torch import Tensor, cat
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Parameter, BatchNorm1d, ReLU, ModuleList, BCEWithLogitsLoss, MSELoss, Embedding
from torch_geometric.data import DataLoader
from torch_geometric.nn.functional import bro, gini
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np

from .molecule_net import MoleculeNetHBonds, CustomDatasetHBonds
from .mol_encoder import AtomEncoder, BondEncoder

cls_criterion = BCEWithLogitsLoss()
reg_criterion = MSELoss()


class MolData(LightningDataModule):
    def __init__(
            self,
            root,
            name,
            path=None,
            smiles_idx=None,
            y_idx=None,
            task_type='regression',
            hydrogen_bonds=False,
            fake=False,
            fake_proba=0.15,
            hbond_cutoff_dist=2.35,
            hbond_top_dists=(4, 5, 6),
            batch_size=32,
    ):
        super().__init__()
        self.name = name
        self.root = root
        self.path = path
        self.smiles_idx = smiles_idx
        self.y_idx = y_idx
        self.hydrogen_bonds = hydrogen_bonds
        self.fake_hbonds = fake
        self.fake_hbonds_proba = fake_proba
        self.hbond_cutoff_dist = hbond_cutoff_dist
        self.hbond_top_dists = hbond_top_dists
        self.batch_size = batch_size
        if self.path is None:  # then it's an OGB dataset
            self.dataset_class = MoleculeNetHBonds
            ogb_name = f'ogbg-mol{name}'
            ogb_dataset = PygGraphPropPredDataset(name=ogb_name, root='/tmp/ogb')
            self.split_dict = ogb_dataset.get_idx_split()
            self.task_type = ogb_dataset.task_type
            self.num_tasks = ogb_dataset.num_tasks
            del ogb_dataset
        else:
            self.dataset_class = CustomDatasetHBonds
            self.num_tasks = len(self.y_idx)
            self.task_type = task_type
        # only need the split idx from the ogb dataset
        if self.name in ['antibiotic']:
            self.task_type = 'classification'
            self.num_tasks = 1
            return

    def setup(self, stage: None):
        if stage in (None, 'fit'):
            self.dataset = self.dataset_class(
                root=self.root,
                raw_path=self.path,
                smiles_idx=self.smiles_idx,
                y_idx=self.y_idx,
                hbonds=self.hydrogen_bonds,
                fake=self.fake_hbonds,
                proba=self.fake_hbonds_proba,
                hbond_cutoff_dist=self.hbond_cutoff_dist,
                hbond_top_dists=self.hbond_top_dists,
                name=self.name,
            )
        if self.name == 'antibiotic' or (self.path is not None and self.task_type == 'classification'):
            sss_tvt = StratifiedShuffleSplit(n_splits=2, test_size=0.20, random_state=0)
            sss_vt = StratifiedShuffleSplit(n_splits=2, test_size=0.50, random_state=0)
            train_index, valtest_index = next(sss_tvt.split(self.dataset.data.y, self.dataset.data.y))
            val_index, test_index = next(sss_vt.split(self.dataset.data.y[valtest_index],
                                                      self.dataset.data.y[valtest_index]))
            val_index = valtest_index[val_index]
            test_index = valtest_index[test_index]
            assert not np.intersect1d(train_index, val_index).any()
            assert not np.intersect1d(train_index, test_index).any()
            assert not np.intersect1d(val_index, test_index).any()
            self.split_dict = {'train': train_index, 'valid': val_index, 'test': test_index}
        elif self.path is not None and self.task_type == 'regression':
            sss_tvt = ShuffleSplit(n_splits=2, test_size=0.20, random_state=0)
            sss_vt = ShuffleSplit(n_splits=2, test_size=0.50, random_state=0)
            train_index, valtest_index = next(sss_tvt.split(self.dataset.data.y, self.dataset.data.y))
            val_index, test_index = next(
                sss_vt.split(self.dataset.data.y[valtest_index], self.dataset.data.y[valtest_index]))
            val_index = valtest_index[val_index]
            test_index = valtest_index[test_index]
            assert not np.intersect1d(train_index, val_index).any()
            assert not np.intersect1d(train_index, test_index).any()
            assert not np.intersect1d(val_index, test_index).any()
            self.split_dict = {'train': train_index, 'valid': val_index, 'test': test_index}

    def train_dataloader(self):
        return DataLoader(
            self.dataset[self.split_dict['train']],
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset[self.split_dict['valid']],
            batch_size=self.batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset[self.split_dict['test']],
            batch_size=self.batch_size,
            num_workers=1,
        )


class GINConv(MessagePassing):
    def __init__(self, emb_dim, h_bonds=False):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr="add")
        self.mlp = Sequential(Linear(emb_dim, 2*emb_dim), BatchNorm1d(2*emb_dim), ReLU(), Linear(2*emb_dim, emb_dim))
        self.eps = Parameter(Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim, hydrogen_bonds=h_bonds)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, h_bonds=False):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = Linear(emb_dim, emb_dim)
        self.root_emb = Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim, hydrogen_bonds=h_bonds)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class Net(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Net')
        parser.add_argument(
            '--initial_learning_rate', type=float, default=0.001
        )
        parser.add_argument(
            '--gnn_type', choices=['gin', 'gcn'], default='gin',
        )
        parser.add_argument(
            '--n_conv_layers', type=int, default=5,
        )
        parser.add_argument(
            '--embedding_dim', type=int, default=300,
        )
        parser.add_argument(
            '--drop_ratio', type=float, default=0.5
        )
        parser.add_argument(
            '--virtual_node', action='store_true', default=False,
        )
        parser.add_argument(
            '--JK', choices=['last', 'sum'], default='last',
        )
        parser.add_argument(
            '--residual', action='store_true', default=False,
        )
        parser.add_argument(
            '--gini', type=float, default=0.0,
            help='Exponent of the Gini constraint. Default: 0.0'
        )
        parser.add_argument(
            '--BRO', type=float, default=None,
            help='BRO regularization paramter. Default: 0.000'
        )
        return parent_parser

    def __init__(self, task_type, h_bonds, num_tasks, evaluator, conf):
        super().__init__()
        self.save_hyperparameters(conf)
        self.n_conv_layers = self.hparams.n_conv_layers
        self.initial_learning_rate = self.hparams.initial_learning_rate
        self.embedding_dim = self.hparams.embedding_dim
        self.drop_ratio = self.hparams.drop_ratio
        self.gnn_type = self.hparams.gnn_type
        self.virtual_node = self.hparams.virtual_node
        self.JK = self.hparams.JK
        self.residual = self.hparams.residual
        self.gini = self.hparams.gini
        self.BRO = self.hparams.BRO

        self.h_bonds = h_bonds
        self.task_type = task_type
        self.num_tasks = num_tasks
        self.evaluator = evaluator

        self.test_preds = None
        self.test_y = None

        self.atom_encoder = AtomEncoder(emb_dim=self.embedding_dim, hydrogen_bonds=self.h_bonds)

        if self.virtual_node:
            self.virtualnode_embedding = Embedding(1, self.embedding_dim)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            self.mlp_virtualnode_list = ModuleList()
            for layer in range(self.n_conv_layers - 1):
                self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(self.embedding_dim, 2 * self.embedding_dim),
                        BatchNorm1d(2 * self.embedding_dim),
                        ReLU(),
                        Linear(2 * self.embedding_dim, self.embedding_dim),
                        BatchNorm1d(self.embedding_dim),
                        ReLU()),
                )

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        if self.gnn_type == 'gin':
            self.conv = GINConv
        elif self.gnn_type == 'gcn':
            self.conv = GCNConv

        for _ in range(self.n_conv_layers):
            self.convs.append(self.conv(self.embedding_dim, self.h_bonds))
            self.batch_norms.append(BatchNorm1d(self.embedding_dim))

        self.graph_pred_linear = Linear(self.embedding_dim, self.num_tasks)

    def forward(self, batched_data):  # noqa
        x, edge_index, edge_attr, batch = (
            batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        )
        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).type_as(edge_index))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.n_conv_layers):
            if self.virtual_node:
                h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.n_conv_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]
            h_list.append(h)

            if self.virtual_node and layer < self.n_conv_layers - 1:
                virtualnode_embedding_temp = global_add_pool(h_list[layer],
                                                             batch) + virtualnode_embedding
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training,
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training)

        if self.JK == 'last':
            self.node_representation = h_list[-1]
        elif self.JK == 'sum':
            self.node_representation = 0
            for layer in range(self.n_conv_layers + 1):
                self.node_representation += h_list[layer]

        # calculate BRO
        if self.BRO is not None and self.BRO > 0.0:
            self.bro_loss = self.BRO / 2 * bro(self.node_representation, batch)

        self.graph_representation = global_mean_pool(self.node_representation, batch)
        return self.graph_pred_linear(self.graph_representation)

    def CAM(self, task=0):
        """Get the class attribution mapping for a task"""
        if task >= len(range(self.num_tasks)):
            raise ValueError(f'{task} must be between 0 and {self.num_tasks - 1}')
        task_index = task
        regression_weights = self.graph_pred_linear.weight[task_index].cpu().detach().numpy()
        GAP_like = self.graph_representation.cpu().detach().numpy()
        w = (regression_weights * GAP_like)
        w = w.reshape(1, -1)
        return w @ abs(self.node_representation.cpu().detach().numpy().T)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.initial_learning_rate)

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        is_labeled = batch.y == batch.y
        if "classification" in self.task_type:
            loss = cls_criterion(output[is_labeled], batch.y[is_labeled])
        else:
            loss = reg_criterion(output[is_labeled], batch.y[is_labeled])
        self.log('Loss/train', loss)

        # if BRO > 0, add the BRO regularization loss
        if self.BRO is not None and self.BRO > 0.0:
            self.log('Loss/bro', self.bro_loss)
            loss += self.bro_loss

        # if Gini > 0, divide by the Gini coefficient
        if self.gini is not None and self.gini > 0.0:
            g = gini(self.graph_pred_linear.weight)
            self.log('Loss/gini', g)
            loss /= g ** self.gini

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        if 'classification' in self.task_type:
            output = F.sigmoid(output)  # not done in the forward pass! only linear!
        y_true = batch.y.view(output.shape).detach().cpu()
        y_pred = output.detach().cpu()
        return {'y_true': y_true, 'y_pred': y_pred}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs) -> None:
        y_true = cat([result['y_true'] for result in outputs], dim=0).numpy()
        y_pred = cat([result['y_pred'] for result in outputs], dim=0).numpy()
        res = self.evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
        self.log(f'{self.evaluator.eval_metric}', res[self.evaluator.eval_metric])

    def test_epoch_end(self, outputs) -> None:
        y_true = cat([result['y_true'] for result in outputs], dim=0).numpy()
        y_pred = cat([result['y_pred'] for result in outputs], dim=0).numpy()
        res = self.evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
        self.test_preds = y_pred
        self.test_y = y_true
        self.log(f'{self.evaluator.eval_metric}/test', res[self.evaluator.eval_metric])
