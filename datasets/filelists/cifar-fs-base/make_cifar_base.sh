# pip3 install gdown
gdown --id 1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI --output cifar100.zip
unzip cifar100.zip -d ./
rm -rf cifar100.zip
python make_json.py
