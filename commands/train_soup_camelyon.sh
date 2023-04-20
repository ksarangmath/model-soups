
python commands/train_soup.py \
  --fname /srv/share4/ksarangmath3/lsr/model-soups/configs/cmln_full_soup.yml \
  --folder /srv/share4/ksarangmath3/lsr/model-soups/submitit/LP_dino_vitb16_camelyon_full_9_models \
  --soup_size 9 \
  --partition short \
  --nodes 1 --tasks-per-node 1 \
  --time 2080 
