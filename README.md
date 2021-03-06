# Project for the Optimal Transport course at ENSAE Paris

This repository contains the code done for the final project of the Optimal Transport course at ENSAE Paris. It has been
done by Mathilde Kaploun (ENSAE Paris), Thomas Doucet (ENSAE Paris) and Chloé Sekkat (ENSAE Paris \& ENS Paris Saclay)

Final grade: 19/20

# Improving GANs using Optimal Transport

## OT-GAN from scratch

The goal is to implement a GAN variant that builds on Optimal Transport following the approach taken by [this paper](https://arxiv.org/abs/1803.05573),
which develops an OT-GAN. 

We compared this OT-GAN with two other GANs: a simple basic vanilla version inspired by the
work of [Goodfellow et al.](https://arxiv.org/pdf/1406.2661) and DCGAN inspired from [Radford et al. paper](https://arxiv.org/abs/1511.06434). 

# Installation 

After creating your virtual environment, please run 

```
pip install -r requirements.txt
```

This project uses Pytorch, please note that you will need to update manually the versions of ``torch``, ``torchaudio`` 
and ``torchvision`` depending on your hardware.

# About the package

This Python package is self-contained. A Colab link is given so that you can see its applications.

<a href="https://colab.research.google.com/drive/1dKk1_YXdoKikM9yV2L9quL5P-Akm9o-G?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> _Improving GANs with optimal transport_

# Pretrained models

All pretrained models are available [here](https://drive.google.com/drive/folders/1RV0xTk7eZnGPDUOxUWnMOA_Cu6wo5zWJ?usp=sharing).

# Replicate experiments

To train the OT-GAN we implemented, you can use the `main_otgan` script. For instance, one can run:

````bash
python main_otgan.py \
        --seed 0 \
        --batch_size 200 \
        --normalize_mnist False \
        --data_path <<CHOSEN_DATA_DIRECTORY>> \
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
