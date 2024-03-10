# A faire avant dans le terminal :
# pip install lsfb-dataset

# Pour lancer le programme depuis le terminal :
# python .\Database\DownloadData.py

from lsfb_dataset import Downloader

downloader = Downloader(
    dataset='isol',
    destination="D:\DataPII",
    splits=['train', 'fold_0', 'fold_2'],
    signers=list(range(0, 10)),
    include_cleaned_poses=False,
    include_raw_poses=False,
    include_videos=True,
    landmarks=['pose', 'left_hand', 'right_hand'],
    skip_existing_files=True,
)

downloader.download()