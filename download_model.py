"""下载作者公开的 CNN 模型，并在使用前验证文件完整性。"""
import hashlib
from pathlib import Path
import sys
import urllib.request


MODEL_NAME = 'chinese_medicine_classifier_simple.h5'
MODEL_URL = (
    'https://github.com/kayaduster607-jpg/chinese-herbal-recognition/'
    'releases/download/cnn-baseline-v1/' + MODEL_NAME
)
EXPECTED_SHA256 = '52cf58e685c82511120d425dd4c48f834a30bebe905482458d8f4fec2bb1381d'
MODEL_PATH = Path('weights') / MODEL_NAME


def file_sha256(path):
    """分块计算摘要，避免将模型整个读入内存。"""
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_model():
    """先下载到临时文件，验证通过后再放到默认路径。"""
    if MODEL_PATH.exists():
        if file_sha256(MODEL_PATH) == EXPECTED_SHA256:
            print(f'模型已存在且校验通过：{MODEL_PATH}')
            return
        raise ValueError(f'目标路径已有不匹配的文件，请自行检查：{MODEL_PATH}')

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix('.download')
    if temporary.exists():
        raise FileExistsError(f'临时文件已存在，请先检查：{temporary}')
    try:
        print('正在下载约 229 MB 的模型，请稍候……', flush=True)
        request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'chinese-herbal-recognition'})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('xb') as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                output.write(chunk)
        if file_sha256(temporary) != EXPECTED_SHA256:
            raise ValueError('模型文件 SHA-256 校验失败，请重新下载。')
        temporary.replace(MODEL_PATH)
        print(f'下载完成并校验通过：{MODEL_PATH}')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        download_model()
    except (OSError, ValueError) as exc:
        sys.exit(f'模型下载失败：{exc}')
