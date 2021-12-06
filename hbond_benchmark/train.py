from argparse import ArgumentParser
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from ogb.graphproppred import Evaluator

from hbond_benchmark.model import Net, MolData


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset_name', type=str, default='hiv')
    parser.add_argument('--dataset_root', type=str, default='data')
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--smiles_idx', type=int, default=None)
    parser.add_argument('--y_idx', type=int, nargs='*', default=None)
    parser.add_argument('--task_type', type=str, default='regression')
    parser.add_argument('--hbonds', action='store_true', default=False)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--early_stopping', type=int, default=None)
    parser.add_argument('--hbond_cutoff_dist', type=float, default=2.35)
    parser.add_argument('--hbond_top_dists', nargs='*', type=int, default=(4, 5, 6))

    parser = Net.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    return parser.parse_args(args)


def train(args):
    mol_data = MolData(
        root=args.dataset_root,
        name=args.dataset_name,
        path=args.dataset_path,
        smiles_idx=args.smiles_idx,
        y_idx=args.y_idx,
        task_type=args.task_type,
        hydrogen_bonds=args.hbonds,
        hbond_cutoff_dist=args.hbond_cutoff_dist,
        hbond_top_dists=args.hbond_top_dists,
        batch_size=args.batch_size,
    )

    if args.dataset_name in ['antibiotic'] or (args.dataset_path is not None and args.task_type == 'classification'):
        # any AUCROC evaluator will do
        evaluator = Evaluator('ogbg-molhiv')
    elif args.dataset_path is not None and args.task_type == 'regression':
        # any OGB regression task evaluator will do
        evaluator = Evaluator('ogbg-molesol')
    else:
        evaluator = Evaluator(f'ogbg-mol{args.dataset_name}')

    for _ in range(args.n_runs):
        if args.early_stopping is not None:
            early_stopping = EarlyStopping(
                monitor=f'{evaluator.eval_metric}',
                mode='min' if 'rmse' in evaluator.eval_metric else 'max',
                patience=args.early_stopping,
            )
            trainer = Trainer.from_argparse_args(args, callbacks=[early_stopping])
        else:
            trainer = Trainer.from_argparse_args(args)

        trainer.checkpoint_callback.monitor = f'{evaluator.eval_metric}'
        trainer.checkpoint_callback.mode = 'min' if 'rmse' in evaluator.eval_metric else 'max'
        trainer.checkpoint_callback.save_top_k = 1
        model = Net(
            task_type=mol_data.task_type,
            h_bonds=args.hbonds,
            num_tasks=mol_data.num_tasks,
            evaluator=evaluator,
            conf=args,
        )

        trainer.fit(model, mol_data)
        trainer.test()
        del model
        del trainer
        if args.early_stopping is not None:
            del early_stopping


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    train(args)
