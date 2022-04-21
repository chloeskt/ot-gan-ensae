# Project for the Optimal Transport course at ENSAE Paris

This repository contains the code done for the final project og the Optimal Transport course at ENSAE Paris. It has been
done by Mathilde Kaploun (ENSAE Paris), Thomas Doucet (ENSAE Paris) and Chloé Sekkat (ENSAE Paris - ENS Paris Saclay)

# Improving GANs using Optimal Transport
## OT-GAN from scratch

The goal is to implement a GAN variant that builds on Optimal Transport following the approach taken by [this paper](https://arxiv.org/abs/1803.05573),
which develops a OT GAN. 

# Installation 

After creating your virtual environment, please run 

```
pip install -r requirements.txt
```

This project uses Pytorch, please note that you will need to update manually the versions of ``torch``, ``torchaudio`` 
and ``torchvision`` depending on your hardware.

# About the package

This Python package is self-contained. A Colab link is given so that you can see its applications.

- [ ] Add final colab link. 

# Pretrained models

All pretrained models are available [here](https://drive.google.com/drive/folders/1RV0xTk7eZnGPDUOxUWnMOA_Cu6wo5zWJ?usp=sharing).

# Replicate experiments

To train the OT-GAN we implemented, you can use the `main_otgan` script. For instance, one can run:

````bash
python main_otgan.py \
        --seed 0 \
        --batch_size 200 \
        --normalize_mnist False \
        --data_path <<CHOSEN_DIRECTORY>> \
        --epochs 200 \
        --patience 10 \
        --latent_dim 50 \
        --latent_type uniform \
        --kernel_size 3 \
        --critic_learning_rate 1e-4 \
        --generator_learning_rate 1e-4 \
        --gen_hidden_dim 256 \
        --critic_hidden_dim 32 \
        --critic_output_dim 8192 \
        --weight_decay 0. \
        --eps_regularization 1. \
        --nb_sinkhorn_iterations 100 \
        --output_dir <<CHOSEN_OUTPUT_DIRECTORY>> \
        --save True \
        --device cuda
````

To see all possible arguments, run:

```bash
python main_otgan.py --help
```

```bash
python main_vanillagan.py --help
```

```bash
python main_dcgan.py --help
```

# Acknowledgments

The ``early_stopping_pytorch`` module has been taken from: https://github.com/Bjarten/early-stopping-pytorch 
