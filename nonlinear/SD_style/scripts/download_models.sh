#!/bin/bash
wget -O models/ldm/celeba256/celeba-256.zip https://ommer-lab.com/files/latent-diffusion/celeba.zip
wget -O models/ldm/ffhq256/ffhq-256.zip https://ommer-lab.com/files/latent-diffusion/ffhq.zip
wget -O models/ldm/lsun_churches256/lsun_churches-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_churches.zip
wget -O models/ldm/lsun_beds256/lsun_beds-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip
wget -O models/ldm/cin256/model.zip https://ommer-lab.com/files/latent-diffusion/cin.zip



cd models/ldm/celeba256
unzip -o celeba-256.zip

cd ../ffhq256
unzip -o ffhq-256.zip

cd ../lsun_churches256
unzip -o lsun_churches-256.zip

cd ../lsun_beds256
unzip -o lsun_beds-256.zip

cd ../cin256
unzip -o model.zip

cd ../..