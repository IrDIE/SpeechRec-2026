import torch
import torch.nn as nn
from utils_plightning import SpeechCommandsDataModule
from utils_plightning import M5, HyperparameterLogger, get_flops, print_flops
import pytorch_lightning as pl
from melbanks import LogMelFilterBanks
from clearml import Task


# Experiment runner function
def run_experiment(experiment_name, n_groups,  data_module, 
                   n_input=1, n_output=2, max_epochs=15, n_mels=80):

    task = Task.init(project_name="Speech_Commands_Comparison",
                     task_name=experiment_name)
    mel_transform = LogMelFilterBanks(
        samplerate=16000,        # Speech Commands sample rate
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
        f_min_hz=0.0,
        f_max_hz=8000.0,         # or None to use samplerate/2
        norm_mel='slaney',       # optional, common choice
        mel_scale='htk'
    )
    # Create model with specific n_groups
    model = M5(
        n_input=n_mels,
        n_output=n_output,
        n_groups=n_groups,
        experiment_name=experiment_name,
        learning_rate=1e-3,
        transform=mel_transform
    )
    
    sample_data, _ = next(iter(data_module.train_dataloader()))
    sample_shape = sample_data.shape
    result_flops = get_flops(model, sample_shape)
    # result_flops.update({'n_groups': n_groups, 'n_mels': n_mels})
    model.hparams.update(result_flops)

    
    # Create TensorBoard logger with experiment name
    logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs",
        name=experiment_name,
        default_hp_metric=False
    )
    # logger.log_dict(result_flops)  # Log FLOPs and hyperparameters to TensorBoard
    for k, v in result_flops.items():
        if k != 'input_shape' and type(v) != str: model.log('hp/' + k, v)  # Log each hyperparameter and FLOP metric to TensorBoard under 'hp/' namespace
    # Create trainer with experiment-specific settings
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        callbacks=[HyperparameterLogger()]  # Add hyperparameter logging
    )
    
    # Train the model
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    task.close()
    
    return model, trainer


# Function to run multiple experiments with different n_groups
def run_comparison_experiments(dataloader,  
                               n_input=1, n_output=2, max_epochs=10):  
    """
    Run multiple experiments with different n_groups values for comparison
    """
    experiments = []
    n_groups_values = [1, 2, 4, 8]  # Different group configurations to test
    n_mels_set = [20, 40, 80]
    
    for n_groups in n_groups_values:
        for n_mel in n_mels_set:
            experiment_name = f"groups_{n_groups}_mels_{n_mel}"
            print(f"\n{'='*50}")
            print(f"Running experiment: {experiment_name} ")
            print(f"{'='*50}")
            
            model, trainer = run_experiment(
                experiment_name=experiment_name,
                n_groups=n_groups,
                data_module=dataloader,
                n_input=n_input,
                n_output=n_output,
                max_epochs=max_epochs,
                n_mels=n_mel,
            )
            
            experiments.append({
                'name': experiment_name,
                'n_groups': n_groups,
                'n_mels': n_mel,
                'model': model,
                'trainer': trainer
            })
    
    return experiments

# Complete training example
if __name__ == "__main__":
    # Configure device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set up data module
    dm = SpeechCommandsDataModule(
        data_dir="/mnt/d/ITMO/2026-SpeechRec/all_data/hw1/",
        batch_size=256,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
        download=False,
    )
    
    # Setup data
    dm.setup(stage="fit")

    


    run_comparison_experiments(dm, n_input=1, n_output=len(dm.labels), max_epochs=2)
