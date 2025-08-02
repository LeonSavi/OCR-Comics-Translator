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
    ## OLD:  return [requests.get(i).content for i in matches]
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


def get_box(coord1: Box, coord2: Optional[Box]=None) -> Box:
    ## NOTE: there could be a better way to do this by looking into what "readtext" returns
    ## still, it's clear and quick as we compare maximum 8 numbers every time
    points = list(coord1)
    if coord2 is not None:
        points.extend(coord2)
    return Box.from_points(points)


def _handle_newline_text(text: str) -> str:
    if text[-1] == "-":
        return text[:-1] 
    else:
        return text


def _cleanup_spanish(text: str) -> str:
    """assumes spanish text"""
    is_bang = (text[0]=='i') and (text[1]!=' ') # it is a '!'
    if is_bang:
        return text[1:] 
    else:
        return text


def text_finding(
    image: bytes, 
    reader: easyocr.Reader,
    manga_lang: str,
    threshold_diff: int
) -> Gen[FoundText]:
    ## again, i'll change the function
    ## we are moving a lot around here, extra care when testing !
    iter_list = read_page(reader, image)
    ## sort boxes vertically
    ## NOTE: maybe use min/max value of the "y" coord instead of the min ?
    iter_list.sort(key=lambda x: x.box.mean().y)
    ## more pythonic but less clear
    while iter_list:
        checked = []
        line_cnt = 0
        ## as you use only the index to initialize values, there is a trick
        it = iter(iter_list)
        ## this is the first iteration
        first = next(it)
        prev_low_mean_bbox = first.box.mean()
        coordinates = get_box(first.box)
        text = _handle_newline_text(first.text)
        line_cnt += 1
        checked.append(first)
        iter_list.remove(first)
        ## and these are all the other iterations
        for item in it:
            bbox = item.box
            mean_bbox = bbox.mean()
            curr_text = item.text
            if prev_low_mean_bbox.is_close_to(mean_bbox, threshold_diff):
                coordinates = get_box(coordinates, bbox)
                text = _handle_newline_text(text) + curr_text
                prev_low_mean_bbox = mean_bbox
                checked.append(item)
                line_cnt += 1
        for item in checked:
            iter_list.remove(item)
        #clean up spanish text
        if manga_lang == "es":
            text = _cleanup_spanish(text)
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
    ##
    ##  -> Union[TextResult, List[TextResult]]:
    ##
    ## https://github.com/DeepLcom/deepl-python/blob/main/deepl/api_data.py#L12
    ##
    ## class TextResult:
    ##     """Holds the result of a text translation request."""
    ##
    ##     def __init__(
    ##         self,
    ##         text: str,
    ##         detected_source_lang: str,
    ##         billed_characters: int,
    ##         model_type_used: Optional[str] = None,
    ##     ):
    ##         self.text = text
    ##         self.detected_source_lang = detected_source_lang
    ##         self.billed_characters = billed_characters
    ##         self.model_type_used = model_type_used
    ##
    ##     def __str__(self):
    ##         return self.text
    ## returns the above

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


def pdf_file(pages: list[Image.Image]) -> BytesIO:
    images = [i.convert('RGB') for i in pages]
    pdf_bytes = BytesIO()
    images[0].save(pdf_bytes, format='PDF', save_all=True, append_images=images[1:])
    pdf_bytes.seek(0)
    return pdf_bytes


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


def translate_img(
    image: bytes, reader: easyocr.Reader, manga_lang: LiteralString, thres: int
) -> Image.Image:
    blocks = text_finding(
        image,
        reader=reader,
        manga_lang=manga_lang,
        threshold_diff=thres
    )
    trad_block = deepL_translate(
        blocks,
        source_l = manga_lang.upper(),
        target_l = "EN-GB"
    )
    img_trans = apply_translation(image, trad_block)
    return img_trans


def save_chapter(location: Path, pages: list[Image.Image]):
    with location.open("wb") as f:
        stream = pdf_file(pages).getvalue()
        f.write(stream)


def translate_chapter(manga_lang: LiteralString, url: str, thres: int) -> Gen[Image.Image]:
    ## get a translator
    reader = easyocr.Reader(
        [manga_lang.lower(), 'en'],
        detector='DB',
        gpu=torch.cuda.is_available()
    ) 
    ## get the images to translate
    img_list = get_images(url)
    ## do the translations
    for image in img_list:
        translation = translate_img(image, reader, manga_lang, thres)
        yield translation


def build_save_location(comic_name: str, chapter_name: str) -> Path:
    comic_folder = ROOT_FOLDER / comic_name
    comic_folder.mkdir(parents=True, exist_ok=True)
    save_location = (comic_folder / chapter_name).with_suffix(".pdf")
    return save_location


def process_chapter(
    manga_lang: LiteralString,
    url: str,
    comic_name: str,
    chapter_name: str,
    thres: int
):
    translated_pages = translate_chapter(manga_lang, url, thres)
    translated_pages = list(translated_pages)

    save_location = build_save_location(comic_name, chapter_name)
    save_chapter(save_location, translated_pages)


def main():

    ## to be params in the future
    manga_lang = 'es'
    url = "https://mangapark.net/title/301153-es_419-hadacamera/9280970-vol-6-ch-50"
    # pixel difference between mean x and y axis of previous text box to following text box to be considered as part of the same sentence.
    ## feels like this should be a global const
    thres = 35 
    comic_name = "hadacamera"
    chapter_name = comic_name+'-'+url[-7::]

    process_chapter(manga_lang, url, comic_name, chapter_name, thres)

    

if __name__ == '__main__':
    main()


