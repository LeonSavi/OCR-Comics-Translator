from io import BytesIO
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
from typing import Iterable, LiteralString, NamedTuple
from typing_extensions import Self

import deepl
from deepL_api import api as DEEPL_API
import easyocr
import language_tool_python
from PIL import Image, ImageDraw, ImageFont
import requests
from requests import HTTPError


class TextCleaner:

    def __init__(self, txt_lang:str):
        
        if not self.is_java_installed():
            self.tool = None # LanguageToolPublicAPI does not work
            print("Please install java to use language-tool-python")
        elif txt_lang.lower() == "jp":
            print('''Japanese Language is not fully supported in language_tool_python.\n
                          \n
                          Y (Use the grammar checker)\n
                          N (Avoid)\n''')
            answer = input("Proceed anyway? (Y/N)").strip().lower()
            if answer == 'y':
                self.tool = language_tool_python.LanguageTool(language=txt_lang)
            else:
                self.tool = None
        else:
            self.tool = language_tool_python.LanguageTool(language=txt_lang)
    
    def _language_tool(self,text:str):
        return language_tool_python.utils.correct(text, self.tool.check(text))
    
    def clean_up(self, text: str | list[str]) -> str | list[str]:
        if not self.is_java_installed():
            return text
        if isinstance(text, list):
            return list(map(self._language_tool, text))
        elif isinstance(text, str):
            return self._language_tool(text)
        raise Exception("unrechable")
    
    @staticmethod
    def is_java_installed() -> bool:
        return shutil.which("java") is not None


class Point(NamedTuple):
    """classic numeric 2D coordinate"""
    x: float
    y: float

    def is_close_to(self, other: Self, threshold: int) -> bool:
        cond1 = abs(self.x - other.x)
        cond2 = abs(self.y - other.y)
        return (cond1 <= threshold) and (cond2 <= threshold)


class Box(NamedTuple):
    """basic bounding box"""
    bottom_left: Point
    bottom_rigth: Point
    top_left: Point
    top_right: Point

    @classmethod
    def from_tuple(cls, t: tuple[Point, Point, Point, Point]) -> Self:
        return cls(*t)

    def center(self) -> Point:
        """returns the point in the middle of the box"""
        mean_x = statistics.mean(p.x for p in self)
        mean_y = statistics.mean(p.y for p in self)
        return Point(mean_x, mean_y)

    @classmethod
    def around(cls, points: list[Point]) -> Self:
        """ from a set of points, get the box that surrounds them all """
        min_x = min(p.x for p in points)
        min_y = min(p.y for p in points)
        max_x = max(p.x for p in points)
        max_y = max(p.y for p in points)
        return cls.from_minmax(min_x, min_y, max_x, max_y)

    @classmethod
    def from_minmax(cls, min_x: float, min_y: float, max_x: float, max_y: float) -> Self:
        return cls(
            Point(min_x, min_y),
            Point(max_x, min_y),
            Point(min_x, max_y),
            Point(max_x, max_y)
        )

    @classmethod
    def default(cls) -> Self:
        return cls(
            Point(0, 0),
            Point(0, 0),
            Point(0, 0),
            Point(0, 0)
        )


class ReadText(NamedTuple):
    """
    custom type for what easyocr.Reader().readtext() returns

    ## SEE
    https://deepwiki.com/JaidedAI/EasyOCR/3-basic-usage
    """
    box: Box
    text: str
    confidence: float

    @classmethod
    def from_easyocr_readtext(cls, ocr_result: tuple) -> Self:
        """
        ocr_result appears to have the points sorted in counterclockwise
        order, but it's not sure so we rely on our constructor for 
        a deterministic order.
        """
        # TODO: readtext returns (x, y) or (y, x) ?
        points, text, confidence = ocr_result
        return cls(Box.around(points), text, confidence)


class FoundText(NamedTuple):
    coordinates: Box
    text: str
    line_count: int


class FormattedText(NamedTuple):
    coordinates: Box
    text: str
    formatted: str


class Chapter(NamedTuple):
    """just a context"""
    manga_lang: LiteralString
    url: str
    comic_name: str
    chapter_name: str


# pixel difference between mean x and y axis of previous text box to following text box to be considered as part of the same sentence.
THRESHOLD_DIFFERENCE = 35
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
ROOT_FOLDER = Path(__file__).parent / "output"
PATTERN_JPEG = re.compile(r'https[^"\']+?\.jpeg')
EN_GB = "EN-GB"


def get_images(url: str) -> Iterable[bytes]:
    response = requests.get(url, headers=HEADERS) 
    # TODO: better error handling
    try:
        response.raise_for_status()
    except HTTPError as e:
        print(e)
        raise
    haystack = response.content.decode(response.encoding or "utf-8")
    matches:list[str] = re.findall(PATTERN_JPEG, haystack)
    for m in matches:
        resp = requests.get(m)
        # TODO: better error handling
        try:
            resp.raise_for_status()
        except HTTPError as e:
            print(e)
            raise
        img = resp.content
        assert isinstance(img, bytes), "not all images were bytes"
        yield img


def handle_newline_text(text1:str, text2:str = '') -> str:
    _text1 = text1.strip()
    if _text1.endswith(("-", "_")):
        return _text1[:-1] + text2
    else:
        return _text1 + ' ' + text2 #add blank if it s not a continuation 

def cleanup_spanish(text: str) -> str:
    """assumes spanish text"""
    if len(text) < 2:
        return text
    # it is a '!'
    if text.startswith("i "):
        return text[1:] 
    else:
        return text


def find_text_in_image(image: bytes, reader: easyocr.Reader, manga_lang: str) -> Iterable[FoundText]:
    iter_list = list(map(
        ReadText.from_easyocr_readtext,
        reader.readtext(image, detail=1)
    ))
    # sort boxes vertically, top to bottom
    iter_list.sort(key=lambda x: x.box.center().y)
    # more pythonic but less clear
    while iter_list:
        checked = []
        line_cnt = 0
        it = iter(iter_list)
        # this is the first iteration
        first = next(it)
        prev_low_mean_bbox = first.box.center()
        coordinates = Box.around(list(first.box))
        text = handle_newline_text(first.text)
        line_cnt += 1
        checked.append(first)
        # and these are all the other iterations
        for item in it:
            bbox = item.box
            mean_bbox = bbox.center()
            curr_text = item.text
            if prev_low_mean_bbox.is_close_to(mean_bbox, THRESHOLD_DIFFERENCE):
                coordinates = Box.around(list(coordinates) + list(bbox))
                text = handle_newline_text(text, curr_text)
                prev_low_mean_bbox = mean_bbox
                checked.append(item)
                line_cnt += 1
        for item in checked:
            iter_list.remove(item)
        # clean up spanish text
        if manga_lang == "es":
            text = cleanup_spanish(text)
        yield FoundText(coordinates, text.lower(), line_cnt)


def split_text(text: str, n_lines: int) -> str:
    words = text.split()
    text_lenght = len(words)
    if text_lenght == 1:
        return text
    # NOTE: maybe use a constant ?
    words_per_line = min(round(text_lenght/n_lines), 2)
    text = ""
    for idx, word in enumerate(words, start=1):
        endl = " \n" if idx % words_per_line == 0 else " "
        text += word + endl
    return text


def translate_img(
    image: bytes,
    reader: easyocr.Reader,
    manga_lang: LiteralString,
    deeplc: deepl.DeepLClient
) -> Image.Image:
    blocks = find_text_in_image(image, reader, manga_lang)
    trad_block = deepL_translate(
        blocks,
        source_l = manga_lang.upper(),
        target_l = EN_GB,
        deeplc = deeplc
    )
    return apply_translation(image, trad_block)


def deepL_translate(
    blocks: Iterable[FoundText], 
    deeplc: deepl.DeepLClient,
    source_l: LiteralString = "ES", 
    target_l: LiteralString = EN_GB
) -> Iterable[FormattedText]:
    # deepl_client.translate_text( returns the following thing
    #  -> Union[TextResult, List[TextResult]]:
    # https://github.com/DeepLcom/deepl-python/blob/main/deepl/api_data.py#L12

    _blocks = list(blocks)
    sentences = TextCleaner(source_l).clean_up([x.text for x in _blocks])
    try:
        traductions = deeplc.translate_text(sentences, source_lang=source_l, target_lang=target_l)

        for (coord, text, n_lines), translated in zip(_blocks, traductions):
            translated_txt = _stringify(TextCleaner(target_l).clean_up(translated.text))
            formatted_trad = split_text(translated_txt, n_lines)
            yield FormattedText(coord, text, formatted_trad)
    except ValueError as e:
        print(e)
        yield FormattedText(Box.default(), "", "")


def _stringify(thing) -> str:
    if isinstance(thing, list):
        return " ".join(map(str, thing))
    else:
        return str(thing)


def ffont_path(preferred_font: str="DejaVuSans.ttf") -> str:
    system = platform.system()
    if system == "Windows":
        font_dir = os.path.join(os.environ['WINDIR'], "Fonts")
        font_path = os.path.join(font_dir, preferred_font)
        return font_path
    elif system == "Linux":
        font_path = "/usr/share/fonts/truetype/dejavu"
        return font_path
    raise Exception(f"unsupported platflorm: {system}")


def apply_translation(image_bytes: bytes, translated_text: Iterable[FormattedText]) -> Image.Image:
    font_path = ffont_path("DejaVuSans.ttf")
    image = Image.open(BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    font_txt = ImageFont.truetype(font_path, 12)

    translated_text = list(translated_text)
    for coordinates, _, formatted in translated_text:
        # TODO: check
        min_x, min_y = coordinates.bottom_left
        max_x, max_y = coordinates.top_right

        draw.rectangle([min_x,min_y,max_x,max_y], fill="white")
        draw.multiline_text(
            xy=coordinates[0],
            # anchor='mm',
            text=formatted,
            fill ="black",
            font=font_txt ,
            align="center"
        )
    return image


def translate_chapter(c: Chapter, reader: easyocr.Reader, deeplc: deepl.DeepLClient) -> Iterable[Image.Image]:
    yield from (
        translate_img(i, reader, c.manga_lang, deeplc) for i in get_images(c.url)
    )


def pdf_bytes(pages: list[Image.Image]) -> BytesIO:
    ## is this still a list of images ?
    images = [i.convert('RGB') for i in pages]
    _pdf_bytes = BytesIO()
    images[0].save(_pdf_bytes, format='PDF', save_all=True, append_images=images[1:])
    _pdf_bytes.seek(0)
    return _pdf_bytes


def save(c: Chapter, translated_pages: list[Image.Image]):
    comic_folder = ROOT_FOLDER / c.comic_name
    comic_folder.mkdir(parents=True, exist_ok=True)
    save_location = (comic_folder / c.chapter_name).with_suffix(".pdf")
    save_location.write_bytes(pdf_bytes(translated_pages).getvalue())


def process_chapter(
    c: Chapter,
    reader: easyocr.Reader=None,
    deeplc: deepl.DeepLClient=None  
):
    # handle default args
    _reader = reader or easyocr.Reader(
        [c.manga_lang.lower(), 'en'], 
        detector='DB',
        gpu=False
    ) 
    _deepl_client = deeplc or deepl.DeepLClient(DEEPL_API)
    translated_pages = list(translate_chapter(c, _reader, _deepl_client))
    save(c, translated_pages)


def main():

    ## get inputs from somewhere
    manga_lang = 'es'
    url = "https://mangapark.net/title/301153-es_419-hadacamera/9280970-vol-6-ch-50"
    comic_name = "hadacamera"
    chapter_name = comic_name + '-' + url[-7::]

    ## use them
    c = Chapter(manga_lang, url, comic_name, chapter_name)
    process_chapter(c)


