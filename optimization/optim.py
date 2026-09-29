import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse
from pathlib import Path

import optuna
from helper import get_vae_config
from optuna import Trial
from train_vae import train_vae
from vae.vae_helper import MODEL_STRINGS


def objective(trial: Trial):
    vae_config = get_vae_config()

    epochs = 1
    dir_path = None
    vals_per_epoch = 1
    traces_to_validate = 1_000_000

    seed = vae_config["seed"]
    trace_data = "Pospischil"
    vae_type = vae_config["vae_type"]
    normalizer_type = vae_config["normalizer_type"]
    train_split = vae_config["train_split"]
    model_params = vae_config["model_params"]

    batch_size = trial.suggest_categorical("batch_size", [2**i for i in range(5, 12)])
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01, log=True)
    latent_size = trial.suggest_int("latent_size", 4, 64)
    # beta_norm = trial.suggest_float("beta_norm", 0.001, 0.1, log=True)
    beta = trial.suggest_float("beta", 0.02, 2.0, log=True)

    print(f"Running trial {trial.number=} in process {os.getpid()}")
    print(f"Trial Params: {trial.params}")

    _, _, val_losses, (val_recon_losses, val_kl_losses) = train_vae(
        batch_size,
        learning_rate,
        epochs,
        latent_size,
        seed,
        beta,
        trace_data,
        MODEL_STRINGS[vae_type],
        normalizer_type,
        dir_path,
        train_split,
        model_params,
        vals_per_epoch,
        traces_to_validate,
        save_on_val=False,
    )
    return val_recon_losses[-1] + val_kl_losses[-1]


def _run_study(study_num: int, n_trials: int):
    study_dir = Path(f"{current_dir}/studies")
    study_dir.mkdir(parents=True, exist_ok=True)
    study_path = study_dir / f"study{study_num}.db"

    study = optuna.create_study(
        study_name="temp_study",
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    print(study.best_params)


def _parse_args() -> dict:
    parser = argparse.ArgumentParser("Optimize a VAE model.")
    parser.add_argument(
        "study_num",
        type=int,
    )
    parser.add_argument(
        "n_trials",
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    _run_study(args.study_num, args.n_trials)
