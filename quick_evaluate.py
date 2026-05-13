import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tspfn.finetuning.tspfn_finetune import TSPFNFinetune

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Print the config to verify (optional)
    # print(OmegaConf.to_yaml(cfg))

    # Set the random seed for reproducibility
    # Access values using dot notation
    torch.manual_seed(cfg.seed)

    # Initialize the TSPFNFinetune model
    # You can pass the whole cfg or specific subsets
    model = TSPFNFinetune(
        model_checkpoint=cfg.model_checkpoint,
        path_to_data=cfg.path_to_data,
        # Example of accessing nested parameters:
        # device=cfg.model.device 
    )

    # Perform evaluation
    evaluation_results = model.evaluate()

    # Print results
    print(f"Evaluation Results: {evaluation_results}")

if __name__ == "__main__":
    main()