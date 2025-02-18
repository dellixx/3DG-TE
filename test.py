import argparse
import os
import torch
from models import GaussTKGE_3D
from datasets import TemporalDataset
from typing import Dict

parser = argparse.ArgumentParser(
    description="Test GaussTKGE_3D Model"
)
parser.add_argument(
    '--dataset', type=str, default='ICEWS14',
    help="Dataset name"
)
parser.add_argument(
    '--model', default='GaussTKGE_3D', type=str,
    help="Model Name"
)
parser.add_argument(
    '--rank', default=1500, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=2000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)

args = parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    Aggregate metrics for missing lhs and rhs
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}

def test_model(model_name, dataset_name, rank, no_time_emb):
    """
    Load trained model and evaluate on the test set.
    """
    # Set paths for dataset and model
    path=f'best_checkpoints/{dataset_name}'
    
    # Load dataset
    dataset = TemporalDataset(dataset_name)
    sizes = dataset.get_shape()

    # Initialize model
    model = {
        'GaussTKGE_3D': GaussTKGE_3D(sizes, rank, no_time_emb=no_time_emb)
    }[model_name]
    model = model.cuda()

    # Load model parameters
    model_file = os.path.join(path, f'{model_name}.pkl')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"Loading model from {model_file}")
    model.load_state_dict(torch.load(model_file))

    # Set model to evaluation mode
    model.eval()

    # Evaluate on test set
    print("Evaluating on test set...")
    results = avg_both(*dataset.eval(model, 'test', -1))
    print("Test Results:")
    print(f"MRR: {results['MRR']}")
    print(f"Hits@[1,3,10]: {results['hits@[1,3,10]']}")

if __name__ == '__main__':
    test_model(
        model_name=args.model,
        dataset_name=args.dataset,
        rank=args.rank,
        no_time_emb=args.no_time_emb
    )
