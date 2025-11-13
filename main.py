"""Main program for training and evaluation."""

import torch
import warnings
import random
import numpy as np
import dgl
import argparse
from .config import Config
from .models import create_model
from .data import DataLoader
from .training import Trainer
from .samplers import create_sampler
import logging
from .utils.logger import setup_logging
import os
from .utils.visualization import get_all_latent_representations, visualize_latent_space, visualize_raw_data

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Traffic Detection Training')
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument('--dataset', type=int, default=Config.DATASET_SELECT,
                      help='Dataset selection (0: MCFP, 1: USTC-TFC2016-master)')
    parser.add_argument('--model', type=int, default=Config.MODELS_SELECT,
                      help='Model selection (0: CombinedModel, 1: GraphSAGE, 2: CNN1D)')
    parser.add_argument('--sampler', type=str, default=Config.SAMPLER_TYPE,
                      help='Sampler type (bandit or neighbor)')
    parser.add_argument('--epoches', type=int, default=Config.EPOCHS,
                      help='Number of training epoches')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                      help='Training batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                      help='Learning rate')
    parser.add_argument('--use_knn', type=int, default=Config.USE_KNN,
                        help="Use k nearest neighbor graph")
    parser.add_argument('--use_session', type=int, default=Config.USE_SESSION_FLOW,
                        help="Use session flow graph")
    parser.add_argument('-b', type=bool, default=Config.BINARY_CLASS,
                        help="Use multi-class classification")
    parser.add_argument('--sl', type=int, default=Config.seq_len)
    parser.add_argument('-k', type=int, default=Config.K,
                        help="k value for KNN graph")
    parser.add_argument('-p', type=int, default=Config.d_packet,
                        )
    return parser.parse_args()

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Update configuration
    Config.DATASET_SELECT = args.dataset
    Config.MODELS_SELECT = args.model
    Config.SAMPLER_TYPE = args.sampler
    Config.EPOCHS = args.epoches
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    Config.USE_KNN = args.use_knn
    Config.USE_SESSION_FLOW  = args.use_session
    Config.seq_len = args.sl
    Config.K = args.k
    Config.BINARY_CLASS = args.b
    Config.d_packet = args.p
    # if "-P" in Config.MODEL_NAMES[Config.MODELS_SELECT]:
    #     dim = Config.d_packet
    # elif Config.SESSION_PREFIXS[Config.prefix_num] is "P_SA":
    #     dim = Config.seq_len*Config.d_packet
    Config.d_session = Config.seq_len*Config.d_packet
    # Set up logger
    setup_logging()
    # Create save directories
    vis_dir = os.path.join(Config.get_data_dir(), f"len_{Config.d_session}\\visualizations")
    raw_vis_dir = os.path.join(Config.get_data_dir(), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(raw_vis_dir, exist_ok=True)
    Config.VIS_DIR = vis_dir
    Config.RAW_VIS_DIR = raw_vis_dir
    # Log training configuration
    training_config = [
        ("Dataset", Config.DATASET_NAMES[Config.DATASET_SELECT]),
        ("Sampler", Config.SAMPLER_TYPE),
        ("Epochs", Config.EPOCHS),
        ("Batch Size", Config.BATCH_SIZE),
        ("Learning Rate", Config.LEARNING_RATE),
        # ("Seed", args.seed),
        ("Device", Config.DEVICE)
    ]
    training_config_str = "  ".join([f"{k}: {v}" for k, v in training_config])
    
    # Set random seed
    set_seed(args.seed)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        # logging.info("Initializing data loader...")
        data_loader = DataLoader()
        g = data_loader.prepare_data()
        
        model = create_model(Config.MODELS_SELECT)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        training_config_str += f"  model: {model.name}"
        start_flag = '\n'+"-"*(9)+"Training Config"+"-"*(9) +'\n'
        end_flag = '\n'+"-"*(len(start_flag)-2)
        logging.info(start_flag+training_config_str+end_flag)

        # Create sampler
        sampler = create_sampler(
            Config.SAMPLER_TYPE,
            fanouts=[10, 10],
            train_mask=g.ndata['train_mask'],
            device=Config.DEVICE
        )
        
        # Create trainer
        trainer = Trainer(model, g, sampler, optimizer)
        
        # Start training
        # Assume test_data is raw feature data, test_labels are corresponding labels
        visualize_raw_data(
            data=g.ndata['feat'][g.ndata['test_mask']].clone().detach().cpu().numpy(),
            labels=g.ndata['label'][g.ndata['test_mask']].clone().detach().cpu().numpy(),
            n_components=Config.OUT_CHANNELS,
            save_path=Config.RAW_VIS_DIR
        )
        trainer.train_epochs()
        
        # Get latent representations of all data and visualize
        logging.info("Generating latent representations...")
        latent_reps, labels = get_all_latent_representations(model, sampler, g)
        
        
        
        # Visualize with PCA
        logging.info("Visualizing with PCA...")
        visualize_latent_space(
            latent_reps, 
            labels,
            method='pca',
            n_components=Config.OUT_CHANNELS,
            save_path=Config.VIS_DIR
        )
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
