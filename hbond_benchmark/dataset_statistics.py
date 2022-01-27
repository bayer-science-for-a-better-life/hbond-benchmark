"""
For the paper but also for the "fake Hbond" experiments we need to know the frequency of molecules with H bonds and the
number of such H bonds in those molecules.
"""
import numpy as np
from hbond_benchmark.model import MolData


def get_hbond_stats_for_dset(dset_name, dataset_root=None, dataset_path=None, smiles_idx=None, y_idx=None):
    """

    Args:
        dset_name: nickname for the dataset. Ex: bbbp, hiv, LOD, ...
        dataset_root: where the raw and processed folders will be stored
        dataset_path: where the original data is stored (compulsory for non-OGB datasets)
        smiles_idx: index of the column containing smiles (compulsory for non-OGB datasets)
        y_idx: index of the column(s) containing the target value(s) (compulsory for non-OGB datasets)

    Returns: the number of molecules in the dataset, the number of molecules with at least 1 intramolecular H bond,
    the percent of molecules with intramolecular Hbonds, the average number of nodes in the molecules of that dataset,
    the average number of edges in the molecules of that dataset

    """

    data_h = MolData(root=dataset_root + 'with_hbonds', name=dset_name, hydrogen_bonds=True, path=dataset_path,
                     smiles_idx=smiles_idx, y_idx=y_idx, fake=False, fake_proba=0.)
    data_h.setup(stage='fit')
    data = MolData(root=dataset_root + 'no_hbonds', name=dset_name, hydrogen_bonds=False, path=dataset_path,
                   smiles_idx=smiles_idx, y_idx=y_idx, fake=False, fake_proba=0.)
    data_h.setup(stage='fit')
    data.setup(stage='fit')
    assert len(data_h.dataset) == len(data.dataset)
    mols_with_hbonds = []
    num_nodes = []
    num_edges = []
    num_hbonds = []
    for i in range(len(data_h.dataset)):
        n_hbonds = (data_h.dataset[i].edge_attr == 5).sum().item() // 2
        num_hbonds.append(n_hbonds)
        if n_hbonds > 0:
            mols_with_hbonds.append(i)
        num_nodes.append(data.dataset[i].x.shape[0])
        num_edges.append((data.dataset[i].num_edges)//2)

    return len(data_h.dataset), len(mols_with_hbonds), float(len(mols_with_hbonds))/len(data_h.dataset), \
           np.mean(num_nodes), np.mean(num_edges)


if __name__ == '__main__':

    for dset in ['lod', 'c1a', 'lmp', 'loo', 'loh']:
        l, hb, ph, mn, me = get_hbond_stats_for_dset(dset,
                                                     dataset_root='/home/ghrbw/Projects/hbond_ryan/data/datasets/',
                                                     smiles_idx=1, dataset_path='/home/ghrbw/', y_idx=[2])
        print(dset)

        print('Number of molecules: {}'.format(l))
        print('Number of molecules with intramolecular Hbonds: {}'.format(hb))
        print('Percent of molecules with intramolecular Hbonds: {:.3f}'.format(ph))
        print('Average number of nodes: {:.1f}'.format(mn))
        print('Average number of edges: {:.1f}'.format(me))
        print('**********')



