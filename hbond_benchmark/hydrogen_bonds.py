from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Lipinski, rdMolTransforms
import numpy as np


def h_bond_dist(mol, conf, donor_idx, donor_h_idx, acceptor_idx, hbond_top_dists=(4, 5, 6)):
    if donor_idx == acceptor_idx:
        return np.nan
    # check topological distance
    t_dist = Chem.rdmolops.GetDistanceMatrix(mol)[donor_h_idx, acceptor_idx]
    if len(hbond_top_dists) > 1:
        if t_dist not in hbond_top_dists:
            return np.nan
    else:
        if t_dist < hbond_top_dists[0]:
            return np.nan
    # check angle
    angle = rdMolTransforms.GetAngleDeg(conf, donor_idx, donor_h_idx,
                                        acceptor_idx)
    if angle < 100 or angle > 180:
        return np.nan
    # return distance
    return conf.GetAtomPosition(donor_h_idx).Distance(
        conf.GetAtomPosition(acceptor_idx))


def get_donor_acceptor_distances(mol, n_conformers=10, hbond_top_dists=(4, 5, 6)):
    donors = [x[0] for x in Lipinski._HDonors(mol)]
    acceptors = [x[0] for x in Lipinski._HAcceptors(mol)]
    # isolate donor hydrogens
    mol_with_h = Chem.AddHs(mol)
    donors_hs = [[y.GetIdx() for y in mol_with_h.GetAtoms()[x].GetNeighbors() if
                  y.GetSymbol() == 'H'] for x in donors]

    r = Chem.EmbedMultipleConfs(mol_with_h, numConfs=n_conformers, randomSeed=42, maxAttempts=3)
    if r == -1:
        raise(ValueError("Couldn't embed"))

    distance_tensor = []
    for i, donor_idx in enumerate(donors):
        distance_tensor.append([])
        for acceptor_idx in acceptors:
            distance_tensor[-1].append([])
            for conf in mol_with_h.GetConformers():
                h_ds = []
                for h_idx in donors_hs[i]:
                    h_ds.append(h_bond_dist(mol_with_h, conf, donor_idx, h_idx,
                                            acceptor_idx, hbond_top_dists))
                h_ds = [x for x in h_ds if x > 0]
                if len(h_ds) > 0:
                    distance_tensor[-1][-1].append(min(h_ds))
                else:
                    distance_tensor[-1][-1].append(np.nan)

    return donors, acceptors, np.array(distance_tensor)
