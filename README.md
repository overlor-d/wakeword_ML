# wakeword

Enregistrer des échantillons du mot que vous voulez wake au moins faire 50 de chaque positifs et négatifs
- positif -> dire le wakeword
- négatif -> dire autre chose qui est semblable

`python3 ./record_wakeword.py`

Entrainer le modèle avec les échantillons récoltés en pointant sur les dossiers avec les échantillons

`python3 train_ovos_lr.py --positive wakeword_data/ovos/positive --negative wakeword_data/ovos/negative`

lancer la reco :

`python3 run_ovos_lr.py --mic 9`