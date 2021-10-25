from fastai.vision.all import *
import fastbook

def download_categories(path, categories, n_images):
  if not path.exists():
    path.mkdir()
    for o in categories:
      dest = (path/o)
      dest.mkdir(exist_ok=True)
      urls = search_images_ddg(f' {o} ', max_images=n_images)
      download_images(dest, urls=urls)

def check_images(path):
  fns = get_image_files(path)
  failed = verify_images(fns)
  failed.map(Path.unlink);