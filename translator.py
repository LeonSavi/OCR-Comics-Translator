
from io import BytesIO
import os
import platform
from typing import (
    Any, 
    Iterable,
    NamedTuple,
    Optional
)

from typing_extensions import LiteralString

## third party libraries
from deepl import DeepLClient
import easyocr
from PIL import Image, ImageDraw, ImageFont

from interfaces import Box, FoundText, FormattedText, Gen, ReadText


# pixel difference between mean x and y axis of previous text box to following text box to be considered as part of the same sentence.
THRESHOLD_DIFFERENCE = 35


class PageTranslator(NamedTuple):
    image: bytes
    reader: easyocr.Reader
    manga_lang: LiteralString
    deepl: DeepLClient

    def translate(self) -> Image.Image:
        blocks = self.find_text_in_image()
        trad_block = deepL_translate(
            self.deepl,
            blocks,
            source_l = self.manga_lang.upper(),
            target_l = "EN-GB"
        )
        return self.apply_translation(trad_block)

    def read_page(self) -> list[ReadText]:
        """ this only adds type hints and freezes the "detail" param """
        # results = [
        #     (
        #         [[700, 288], [748, 288], [748, 346], [700, 346]], # 4 corners of the box, in a list
        #         '2',
        #         0.35551235377397816),
        #     ... # you get a list of these
        # ]
        # need this step to sort the received points in the correct order
        return list(map(
            ReadText.from_easyocr_readtext,
            self.reader.readtext(self.image, detail=1)
        ))

    def find_text_in_image(self) -> Gen[FoundText]:
        ## NOTE: we are moving a lot around here, extra care when testing !
        iter_list = self.read_page()
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
            if self.manga_lang == "es":
                text = cleanup_spanish(text)
            yield FoundText(coordinates, text.lower(), line_cnt)

    def apply_translation(self, translated_text: Iterable[FormattedText]) -> Image.Image:
        font_path = ffont_path("DejaVuSans.ttf")
        image = Image.open(BytesIO(self.image))
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


def deepL_translate(
    deepl_client: DeepLClient,
    blocks: Iterable[FoundText], 
    source_l: Optional[LiteralString] = "ES", 
    target_l: Optional[LiteralString] = "EN-GB"
) -> Gen[FormattedText]:
    ## deepl_client.translate_text( returns the following thing
    ##  -> Union[TextResult, List[TextResult]]:
    ## https://github.com/DeepLcom/deepl-python/blob/main/deepl/api_data.py#L12


    # to move below?  
    ## we are actually calling the API once per page
    ## having a single client live for the whole 
    ## chapter makes more sense, 
    for coord, text, n_lines in blocks:
        traduction = deepl_client.translate_text(
            text,
            source_lang=source_l,
            target_lang=target_l
        )
        formatted_trad = split_text(strigify(traduction), n_lines)
        yield FormattedText(coord, text, formatted_trad)


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


def strigify(thing: list[Any] | Any) -> str:
    ## "traduction" is not really a str but it's treated as such.
    ## so I suggest an explicit conversion, also to handle the fact that
    ## it may return list OR str
    if isinstance(thing, list):
        return " ".join(map(str, thing))
    else:
        return str(thing)


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


def surrounding_box(coord1: Box, coord2: Optional[Box]=None) -> Box:
    ## NOTE: there could be a better way to do this by looking into what
    ## "readtext" returns still, it's clear and quick as we compare maximum 8
    ## numbers every time

    # if there is a single box, 
    # the sourrounding box it's the provided one itself
    if coord2 is None:
        return coord1
    points = list(coord1)
    points.extend(coord2)
    return Box.from_points(points)


def handle_newline_text(text: str) -> str:
    return text.removesuffix("-")


def cleanup_spanish(text: str) -> str:
    """assumes spanish text"""
    if len(text) < 2:
        return text
    is_bang = (text[0]=='i') and (text[1]!=' ') # it is a '!'
    if is_bang:
        return text[1:] 
    else:
        return text

