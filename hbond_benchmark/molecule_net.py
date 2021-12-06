import os
import os.path as osp
import shutil
import re
import warnings

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)
from tqdm import tqdm

from .hydrogen_bonds import get_donor_acceptor_distances


x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'HYDROGEN',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


class MoleculeNetHBonds(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.ai/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace', 0, 2],
        'bbbp': ['BBPB', 'BBBP.csv', 'BBBP', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21', -1,
                  slice(0, 12)],
        'toxcast':
        ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data', 0,
         slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox', 0,
                    slice(1, 3)],
        'antibiotic': ['antibiotic', '1-s2.0-S0092867420301021-mmc1.xlsx', 'antibiotic', 2, -1]
        # https://ars.els-cdn.com/content/image/1-s2.0-S0092867420301021-mmc1.xlsx
    }

    def __init__(self, root, name, raw_path=None, smiles_idx=None, y_idx=None,
                 hbonds=True,
                 hbond_cutoff_dist=2.35,
                 hbond_top_dists=(4, 5, 6),
                 transform=None, pre_transform=None,
                 pre_filter=None):  # useless attributes are needed for harmonized signature across data classes
        self.name = name.lower()
        self.hbonds = hbonds
        self.hbond_cutoff_dist = hbond_cutoff_dist
        self.hbond_top_dists = hbond_top_dists
        assert self.name in self.names.keys()
        super(MoleculeNetHBonds, self).__init__(root, transform, pre_transform,
                                                pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        if self.name in ['antibiotic']:
            url = 'https://ars.els-cdn.com/content/image/1-s2.0-S0092867420301021-mmc1.xlsx'
            os.system(f'wget {url} --no-check-certificate --directory-prefix={self.raw_dir}')
            df = pd.read_excel(osp.join(self.raw_dir, self.names[self.name][1]), sheet_name='S1B', skiprows=1)
            df[self.name] = 0
            df[self.name][df.Activity == 'Active'] = 1
            df.to_csv(os.path.join(self.raw_dir, f'{self.names[self.name][2]}.csv'))
        else:
            url = self.url.format(self.names[self.name][1])
            path = download_url(url, self.raw_dir)
            if self.names[self.name][1][-2:] == 'gz':
                extract_gz(path, self.raw_dir)
                os.unlink(path)

    def process(self): # noqa
        from rdkit import Chem

        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in tqdm(dataset):
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            xs = get_atom_features(mol)
            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

            edge_indices, edge_attrs = get_edge_features(mol)

            # get hydrogen bonds
            if self.hbonds:
                edge_indices, edge_attrs = handle_hbonds(mol, smiles, edge_indices, edge_attrs, self.hbond_top_dists,
                                                         self.hbond_cutoff_dist)

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))


class CustomDatasetHBonds(InMemoryDataset):
    r"""We want to be able to use our own data (Bayer or other) and run experiments on them.

        Args:
            root (string): Root directory where the dataset should be saved.
            raw_path (string): full path to where the raw data (csv format) can be found
            name (string): The name of the dataset
            smiles_idx (int): The index of the column containing the smiles string
            y_idx (list): The indices of the columns containing the outcome to predict
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
        """

    def __init__(self, root, name, raw_path, smiles_idx, y_idx, hbonds=True, hbond_cutoff_dist=2.35,
                 hbond_top_dists=(4, 5, 6), transform=None, pre_transform=None, pre_filter=None):
        self.name = name.lower()
        self.raw_path = raw_path
        self.hbonds = hbonds
        self.smiles_idx = smiles_idx
        self.y_idx = y_idx
        self.hbond_cutoff_dist = hbond_cutoff_dist
        self.hbond_top_dists = hbond_top_dists
        assert osp.exists(raw_path)
        super(CustomDatasetHBonds, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return osp.basename(self.raw_path)

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # copy file into raw_dir
        shutil.copy(str(self.raw_path), str(self.raw_dir))

    def process(self): # noqa
        from rdkit import Chem

        with open(self.raw_path, 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        sep = infer_separator(dataset[0])
        data_list = []
        for line in tqdm(dataset):
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(sep)
            smiles = line[self.smiles_idx]
            ys = np.array(line)[self.y_idx]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            try:
                xs = get_atom_features(mol)
            except ValueError:  # weird chemistry, does not fit with prerequisites
                continue
            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

            edge_indices, edge_attrs = get_edge_features(mol)

            # get hydrogen bonds
            if self.hbonds:
                edge_indices, edge_attrs = handle_hbonds(mol, smiles, edge_indices, edge_attrs, self.hbond_top_dists,
                                                         self.hbond_cutoff_dist)

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))


def get_atom_features(mol):
    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(
            str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)
    return xs


def get_edge_features(mol):
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]
    return edge_indices, edge_attrs


def handle_hbonds(mol, smiles, edge_indices, edge_attrs, hbond_top_dists, hbond_cutoff_dist):
    try:
        donors, acceptors, distances = get_donor_acceptor_distances(mol, hbond_top_dists=hbond_top_dists)
        if distances.size > 0:
            distances = np.nanmean(distances, axis=-1)
            for row_idx, donor_idx in enumerate(donors):
                for col_idx, acceptor_idx in enumerate(acceptors):
                    if distances[row_idx][col_idx] <= hbond_cutoff_dist:
                        e = []
                        e.append(e_map['bond_type'].index('HYDROGEN'))
                        e.append(e_map['stereo'].index('STEREONONE'))
                        e.append(e_map['is_conjugated'].index(False))

                        edge_indices += [[donor_idx, acceptor_idx], [acceptor_idx, donor_idx]]
                        edge_attrs += [e, e]
        return edge_indices, edge_attrs
    except ValueError:
        warnings.warn(f'Couldn\'t embed {smiles}, skipping H-bonds.')
        return edge_indices, edge_attrs


def infer_separator(line):
    if len(line.split(',')) > 1:
        return ','
    elif len(line.split('\t')) > 1:
        return '\t'
    else:
        return ';'
