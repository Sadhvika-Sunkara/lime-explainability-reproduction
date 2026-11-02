"""
Downloads the three public datasets into data/ (run once, ~100 MB):

  data/sorted_data_acl/{books,dvd}/{positive,negative}.review
        Multi-Domain Sentiment Dataset v1 (Blitzer, Dredze & Pereira, 2007)
        https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz
  data/imdb/IMDB-Dataset.csv
        IMDB Large Movie Review Dataset (Maas et al., 2011), 50k reviews as one CSV
        (mirror of the Kaggle "IMDB Dataset of 50K Movie Reviews" file)
  data/raw/processed.cleveland.data
        UCI Heart Disease, Cleveland subset (Janosi et al., 1988), CC BY 4.0
        (already included in the repository; re-downloaded only if missing)

Usage:  python scripts/get_data.py
"""
import hashlib
import io
import os
import tarfile
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
BLITZER = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz'
IMDB = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
UCI = 'https://archive.ics.uci.edu/static/public/45/heart+disease.zip'
CLEVELAND_MD5 = '2d91a8ff69cfd9616aa47b59d6f843db'


def fetch(url):
    print('downloading', url)
    with urllib.request.urlopen(url) as r:
        return r.read()


def main():
    d = os.path.join(ROOT, 'sorted_data_acl')
    if not os.path.exists(os.path.join(d, 'dvd', 'positive.review')):
        with tarfile.open(fileobj=io.BytesIO(fetch(BLITZER))) as t:
            for m in t.getmembers():
                parts = m.name.split('/')
                if len(parts) >= 3 and parts[-2] in ('books', 'dvd') and parts[-1] in ('positive.review', 'negative.review'):
                    os.makedirs(os.path.join(d, parts[-2]), exist_ok=True)
                    with open(os.path.join(d, parts[-2], parts[-1]), 'wb') as f:
                        f.write(t.extractfile(m).read())
    p = os.path.join(ROOT, 'imdb', 'IMDB-Dataset.csv')
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'wb').write(fetch(IMDB))
    p = os.path.join(ROOT, 'raw', 'processed.cleveland.data')
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fetch(UCI))) as z:
            open(p, 'wb').write(z.read('processed.cleveland.data'))
    assert hashlib.md5(open(p, 'rb').read()).hexdigest() == CLEVELAND_MD5, 'unexpected Cleveland file'
    print('data ready in', os.path.abspath(ROOT))


if __name__ == '__main__':
    main()
