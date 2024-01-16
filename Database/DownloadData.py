# A faire avant dans le terminal :
# pip install lsfb-datase

# Pour lancer le programme depuis le terminal :
# python .\Database\DownloadData.py

from lsfb_dataset import Downloader

downloader = Downloader(dataset='isol', destination="./Database/Data", include_videos=True)
downloader.download()