# モジュールのインポート
from google_images_download import google_images_download  # モジュールのインポート

# responseオブジェクトの生成
response = google_images_download.googleimagesdownload()

# 検索キーワード/ダウンロード画像の数
arguments = {"keywords": "横浜", "limit": 100, "format": "jpg"}

# ダウンロードの実行
response.download(arguments)
