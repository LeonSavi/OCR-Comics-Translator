# IMPORTANT NOTE: the intent is only refactoring, no adding new functionalities
# look for (##) for **personal style** comments

## standard library
from io import BytesIO
import os
import platform
from pathlib import Path
import re
from typing import Any, Generator, Iterable, LiteralString, Optional, TypeVar

## just for typing convenience
T = TypeVar("T")
type Gen[T] = Generator[T, Any, None]

## third party libraries
from bs4 import BeautifulSoup
import deepl
from deepL_api import api as DEEPL_API
import easyocr
from PIL import Image, ImageDraw, ImageFont
import requests
from requests import HTTPError
import torch

## other project files
## (this time is mostly for type hints and some helpers)
from interfaces import Box, FoundText, FormattedText, read_page


## constants are ALL_CAPS
# pixel difference between mean x and y axis of previous text box to following text box to be considered as part of the same sentence.
THRESHOLD_DIFFERENCE = 35
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
ROOT_FOLDER = Path(__file__).parent / "output"
## compile regex if expression is constant
PATTERN_JPEG = re.compile(r'https[^"\']+?\.jpeg')


def get_images(url: str) -> Gen[bytes]:
    response = requests.get(url, headers=HEADERS) 
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


def surrounding_box(coord1: Box, coord2: Optional[Box]=None) -> Box:
    ## NOTE: there could be a better way to do this by looking into what "readtext" returns
    ## still, it's clear and quick as we compare maximum 8 numbers every time

    # if there is a single box, the sourrounding box it's the provided one itself
    if coord2 is None:
        return coord1
    points = list(coord1)
    points.extend(coord2)
    return Box.from_points(points)


def handle_newline_text(text: str) -> str:
    return text.removesuffix("-")


def cleanup_spanish(text: str) -> str:
    """assumes spanish text"""
    is_bang = (text[0]=='i') and (text[1]!=' ') # it is a '!'
    if is_bang:
        return text[1:] 
    else:
        return text


def find_text_in_image(image: bytes, reader: easyocr.Reader, manga_lang: str) -> Gen[FoundText]:
    ## NOTE: we are moving a lot around here, extra care when testing !
    iter_list = read_page(reader, image)
    ## sort boxes vertically
    ## NOTE: maybe use min/max value of the "y" coord instead of the min ?
    iter_list.sort(key=lambda x: x.box.center().y)
    ## more pythonic but less clear
    while iter_list:
        checked = []
        line_cnt = 0
        ## as you use only the index to initialize values, there is a trick
        it = iter(iter_list)
        ## this is the first iteration
        first = next(it)
        prev_low_mean_bbox = first.box.center()
        coordinates = surrounding_box(first.box)
        text = handle_newline_text(first.text)
        line_cnt += 1
        checked.append(first)
        ## and these are all the other iterations
        for item in it:
            bbox = item.box
            mean_bbox = bbox.center()
            curr_text = item.text
            if prev_low_mean_bbox.is_close_to(mean_bbox, THRESHOLD_DIFFERENCE):
                coordinates = surrounding_box(coordinates, bbox)
                text = handle_newline_text(text) + curr_text
                prev_low_mean_bbox = mean_bbox
                checked.append(item)
                line_cnt += 1
        for item in checked:
            iter_list.remove(item)
        #clean up spanish text
        if manga_lang == "es":
            text = cleanup_spanish(text)
        yield FoundText(coordinates, text.lower(), line_cnt)


def split_text(text: str, n_lines: int) -> str:
    words = text.split()
    text_lenght = len(words)
    if text_lenght == 1:
        return text
    ## NOTE: maybe a constant ?
    words_per_line = min(round(text_lenght/n_lines), 2)
    text = ''
    for idx, word in enumerate(words, start=1):
        text += word
        if idx % words_per_line == 0:
            text += " \n"
        else: 
            text += " "
    return text


def deepL_translate(
    blocks: Iterable[FoundText], 
    source_l: Optional[LiteralString] = "ES", 
    target_l: Optional[LiteralString] = "EN-GB"
) -> Gen[FormattedText]:
    ## deepl_client.translate_text( returns the following thing
    ##  -> Union[TextResult, List[TextResult]]:
    ## https://github.com/DeepLcom/deepl-python/blob/main/deepl/api_data.py#L12
    deepl_client = deepl.DeepLClient(DEEPL_API) #to move below? 
    for coord, text, n_lines in blocks:
        traduction = deepl_client.translate_text(
            text,
            source_lang=source_l,
            target_lang=target_l
        )
        # fake translation
        ## "traduction" is not really a str but it's treated as such.
        ## so I suggest an explicit conversion, also to handle the fact that
        ## it may return list OR str
        if isinstance(traduction, list):
            traduction_ = " ".join(map(str, traduction))
        else:
            traduction_ = str(traduction)
        formatted_trad = split_text(traduction_, n_lines)
        yield FormattedText(coord, text, formatted_trad)


def apply_translation(image_bytes: bytes, translated_text: Iterable[FormattedText]) -> Image.Image:
    font_path = ffont_path("DejaVuSans.ttf")
    image = Image.open(BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    font_txt = ImageFont.truetype(font_path, 13)
    for coordinates, _, formatted in translated_text:
        ## IMPORTANT
        ## TODO: coordinates is most surely wrong here now
        draw.rectangle(coordinates, fill="white")
        draw.multiline_text(
            xy=coordinates[0],
            # anchor='mm',
            text=formatted,
            fill ="black",
            font=font_txt ,
            align="center"
        )
    return image


def pdf_bytes(pages: list[Image.Image]) -> BytesIO:
    ## is this still a list of images ?
    images = [i.convert('RGB') for i in pages]
    _pdf_bytes = BytesIO()
    images[0].save(_pdf_bytes, format='PDF', save_all=True, append_images=images[1:])
    _pdf_bytes.seek(0)
    return _pdf_bytes


def ffont_path(preferred_font="DejaVuSans.ttf") -> str:
    system = platform.system()
    if system == "Windows":
        font_dir = os.path.join(os.environ['WINDIR'], "Fonts")
        font_path = os.path.join(font_dir, preferred_font)
        return font_path
    elif system == "Linux":
        font_path = "/usr/share/fonts/truetype/dejavu"
        return font_path
    raise Exception(f"unsupported platflorm: %s", system)


def translate_img(image: bytes, reader: easyocr.Reader, manga_lang: LiteralString) -> Image.Image:
    blocks = find_text_in_image(image, reader, manga_lang)
    trad_block = deepL_translate(
        blocks,
        source_l = manga_lang.upper(),
        target_l = "EN-GB"
    )
    return apply_translation(image, trad_block)


def save_chapter(location: Path, pages: list[Image.Image]):
    with location.open("wb") as f:
        stream = pdf_bytes(pages).getvalue()
        f.write(stream)


def translate_chapter(manga_lang: LiteralString, url: str) -> Gen[Image.Image]:
    reader = easyocr.Reader(
        [manga_lang.lower(), 'en'],
        detector='DB',
        gpu=torch.cuda.is_available()
    ) 
    for image in get_images(url):
        yield translate_img(image, reader, manga_lang)


def build_save_location(comic_name: str, chapter_name: str) -> Path:
    comic_folder = ROOT_FOLDER / comic_name
    comic_folder.mkdir(parents=True, exist_ok=True)
    save_location = (comic_folder / chapter_name).with_suffix(".pdf")
    return save_location


def process_chapter(manga_lang: LiteralString, url: str, comic_name: str, chapter_name: str):
    translated_pages = list(translate_chapter(manga_lang, url))
    save_location = build_save_location(comic_name, chapter_name)
    save_chapter(save_location, translated_pages)


def main():
    ## get inputs from somewhere
    manga_lang = 'es'
    url = "https://mangapark.net/title/301153-es_419-hadacamera/9280970-vol-6-ch-50"
    comic_name = "hadacamera"
    chapter_name = comic_name+'-'+url[-7::]
    ## use them
    process_chapter(manga_lang, url, comic_name, chapter_name)


if __name__ == '__main__':
    main()

