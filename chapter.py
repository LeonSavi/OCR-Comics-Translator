
from io import BytesIO
from pathlib import Path
import re
from typing import LiteralString, NamedTuple

from bs4 import BeautifulSoup
from deepl import DeepLClient
from deepL_api import api as DEEPL_API
import easyocr
from PIL import Image
import requests
from requests import HTTPError
import torch

from interfaces import Gen
from translator import PageTranslator

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
ROOT_FOLDER = Path(__file__).parent / "output"
PATTERN_JPEG = re.compile(r'https[^"\']+?\.jpeg')


class Chapter(NamedTuple):
    """
    represents a chapter and the tools needed to process it

    reader and client are optional
    """
    language: LiteralString
    url: str
    comic: str
    name: str
    reader: easyocr.Reader
    deepl_client: DeepLClient

    def translate(self) -> Gen[Image.Image]:
        for image in self.get_images():
            pt = PageTranslator(image, self.reader, self.language, self.deepl_client)
            yield pt.translate()

    def process(self):
        self.save(list(self.translate()))

    def build_save_location(self) -> Path:
        comic_folder = ROOT_FOLDER / self.comic
        comic_folder.mkdir(parents=True, exist_ok=True)
        save_location = (comic_folder / self.name).with_suffix(".pdf")
        return save_location

    def save(self, pages: list[Image.Image]):
        with self.build_save_location().open("wb") as f:
            stream = self.pdf_bytes(pages).getvalue()
            f.write(stream)

    @staticmethod
    def pdf_bytes(pages: list[Image.Image]) -> BytesIO:
        ## is this still a list of images ?
        images = [i.convert('RGB') for i in pages]
        _pdf_bytes = BytesIO()
        images[0].save(_pdf_bytes, format='PDF', save_all=True, append_images=images[1:])
        _pdf_bytes.seek(0)
        return _pdf_bytes

    def get_images(self) -> Gen[bytes]:
        response = requests.get(self.url, headers=HEADERS) 
        ## TODO: better error handling
        try:
            response.raise_for_status()
        except HTTPError as e:
            print(e)
            raise
        ## is prettify really needed ? we immediately search the str 
        soup = BeautifulSoup(response.content, features="html.parser")
        haystack = soup.prettify()
        ## be explicit in what you expect to find
        matches:list[str] = re.findall(PATTERN_JPEG, haystack)
        for m in matches:
            resp = requests.get(m)
            ## TODO: better error handling
            try:
                resp.raise_for_status()
            except HTTPError as e:
                print(e)
                raise
            img = resp.content
            ## as you rely on this later, it's better to check it now 
            ## (and maybe handle it in the appropriate way ?)
            assert isinstance(img, bytes), "not all images were strings"
            yield img


def process_chapter(
    language: LiteralString,
    url: str,
    comic: str,
    name: str,
    *,
    reader: easyocr.Reader | None,
    deepl_client: DeepLClient | None
):

    provided_reader = reader or easyocr.Reader(
        [language.lower(), 'en'],
        detector='DB',
        gpu=torch.cuda.is_available()
    )    
    provided_deepl = deepl_client or DeepLClient(DEEPL_API)

    c = Chapter(language, url, comic, name, provided_reader, provided_deepl)
    c.process()


