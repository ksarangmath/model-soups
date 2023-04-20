
python commands/train_soup.py \
  --fname /srv/share4/ksarangmath3/lsr/model-soups/configs/full_soup.yml \
  --folder /srv/share4/ksarangmath3/lsr/model-soups/submitit/LP_clip_vitb16_IN_full_9_models \
  --soup_size 9 \
  --partition long \
  --nodes 1 --tasks-per-node 1 \
  --time 10080
