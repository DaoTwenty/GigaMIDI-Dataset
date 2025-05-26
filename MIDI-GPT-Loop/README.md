# Loop Generation
## Generating and evaluating loops using MIDI-GPT

Here we present the scripts for loop generation as well as evaluation scripts used for our paper. 

### MMM

This repository is the library to run and train our MIDI-GPT-Loop model. It is a fork of the `MMM` repository from the `Metacreation Lab` adapted for training and inference with loops. This library also uses a fork of `MidiTok` allowing the integration of loop tokens.

The main entry point is `inference.py`, where `generate` and `generate_batch`, may be used for MIDI generation

### MMM-Loop

This library holds the scripts used to generate the synthetic MIDI data, and evaluate the accuracy of the generation NOMML controls and the generated loops.

## Usage

> **Note:** The following scripts were created for computing on [Compute Canada](https://docs.alliancecan.ca/). Therefore, slight modifications (file organization, module imports, environment variables) may be needed in order to run these scripts elsewhere.

1. Load the correct modules and install dependencies:

    ```bash
    cd MMM-Loop
    bash slurm/install.sh
    ```

2. Download the model and place it in ``MMM-Loop/models``

3. For loop generation, modify the arguments and run:

    ```bash
    cd MMM-Loop
    sbatch slurm/gen_synthetic_data.sh
    ```

    or 

    ```bash
    cd MMM-Loop
    bash slurm/gen_synthetic_data.sh
    ```

4. For evaluation, modify arguments and run:

    ```bash
    cd MMM-Loop
    sbatch slurm/evaluatet.sh
    ```
