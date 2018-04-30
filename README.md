# Wasserstein-GAN
## Machine Learning Project
## Computer Science master Project for the Machine Learning Course
## Spring 2018
## L.Béthune, G.Coiffier

usage: main.py [-h] [--train] [--noise_dim NOISE_DIM] [--data_dim DATA_DIM]
               [--nb_epoch NB_EPOCH] [--batch_size BATCH_SIZE]
               [--n_batch_per_epoch N_BATCH_PER_EPOCH]
               [--eta_critic ETA_CRITIC] [--clipping CLIPPING]

optional arguments:
  -h, --help            show this help message and exit

  --train, -t           run training phase

  --noise_dim NOISE_DIM, -nd NOISE_DIM

  --data_dim DATA_DIM, -dd DATA_DIM

  --nb_epoch NB_EPOCH, -n NB_EPOCH
                        Number of epochs

  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size

  --n_batch_per_epoch N_BATCH_PER_EPOCH, -nb N_BATCH_PER_EPOCH
                        Number of batch per epochs

  --eta_critic ETA_CRITIC
                        Number of iterations of discriminator per iteration of
                        generator

  --clipping CLIPPING, -c CLIPPING
