# IMPORTANT NOTE: the intent is only refactoring, no adding new functionalities
# look for (##) for **personal style** comments

## standard library
from collections import defaultdict
from copy import deepcopy
from io import BytesIO
import os
import platform
from pathlib import Path
from pprint import pprint
import re
from typing import Any, Generator

## third party libraries
from bs4 import BeautifulSoup
import deepl
from deepL_api import api as DEEPL_API
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import torch


## constants are ALL_CAPS
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

## compile regex if expression is constant
## how about a better name ?
PATTERN = re.compile(r'https[^"\']+?\.jpeg')


def get_images(url: str) -> Generator[bytes, Any, None]:
    response = requests.get(url, headers=HEADERS) 
    ## handle any errors
    response.raise_for_status()
    ## is prettify really needed ? we immediately search the str 
    soup = BeautifulSoup(response.content, features="html.parser")
    haystack = soup.prettify()
    ## be explicit in what you expect to find
    matches:list[str] = re.findall(PATTERN, haystack)
    ## OLD:  return [requests.get(i).content for i in matches]
    images = []
    for m in matches:
        resp = requests.get(m)
        resp.raise_for_status()
        img = resp.content
        ## as you rely on this later, it's better to check it now 
        ## (and maybe handle it in the appropriate way ?)
        assert isinstance(img, bytes), "not all images were strings"
        yield img


def get_box(coord1, coord2=None):

    if coord2 == None:
        points = coord1
    else:
        points = coord1 + coord2

    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    return [(min_x,min_y),(max_x,max_y)]


def coord_list(results: list):
    dict_res = defaultdict(tuple)

    ## https://deepwiki.com/JaidedAI/EasyOCR/3-basic-usage
    # for (bbox, text, prob) in results:
    #     # Get the top-left and bottom-right coordinates
    #     (top_left, top_right, bottom_right, bottom_left) = bbox
    #     top_left = tuple(map(int, top_left))
    #     bottom_right = tuple(map(int, bottom_right))


    for i in results:
        coord = i[0]
        mean_x = np.mean([x[0] for x in coord])
        mean_y = np.mean([y[1] for y in coord])
        dict_res[(mean_x, mean_y)] = i

    dict_res = sorted(dict_res.items(), key= lambda x: x[0][1])
    return dict_res


def text_finding(
    image: bytes,
    reader: easyocr.Reader,
    manga_lang: str,
    threshold_diff: int
):
    
    ## HERE
    ## need to know this type to go on
    results = reader.readtext(image, detail=1)

    iter_list = coord_list(results)

    blocks = []

    while len(iter_list)>0:

        checked = []
        coordinates = []
        text = ''
        line_cnt = 0

        for idx, item in enumerate(iter_list):  # convert to list for stable iteration

            mean_bbox =  item[0]
            bbox = item[1][0]
            curr_text = item[1][1]

            if idx == 0:
                prev_low_mean_bbox = deepcopy(mean_bbox)
                coordinates = get_box(bbox)
                text += curr_text
                line_cnt += 1
                checked.append(item)
            else:
                cond1 = abs(prev_low_mean_bbox[0] - mean_bbox[0])
                cond2 = abs(prev_low_mean_bbox[1] - mean_bbox[1])

                if (cond1 <= threshold_diff) and (cond2 <= threshold_diff): 
                    coordinates = get_box(coordinates, bbox)
                    if (text[-1] == '-'):
                        text = text[:-1] + curr_text
                    else:
                        text += ' ' + curr_text
                    prev_low_mean_bbox = deepcopy(mean_bbox)
                    checked.append(item)
                    line_cnt += 1

        for item in checked:
            iter_list.remove(item)

        #clean up spanish text
        if manga_lang == "es":
            cond = (text[0]=='i') and (text[1]!=' ') #it is a '!'
            if cond:
                text=text[1:] 
        
        blocks.append((coordinates, text.lower(), line_cnt))
    
    return blocks


def split_text(text, n_lines):

    splitted = text.split()
    text_lenght = len(splitted)
    
    if text_lenght == 1:
        return text
    
    n_words = min(round(text_lenght/n_lines),2)
    text = ''

    for idx,word in enumerate(splitted,start=1):
        text += word
        if (idx%n_words)==0:
            text += " \n"
        else: text += " "

    return text



def deepL_translate(blocks, source_l = "ES", target_l = "EN-GB"):

    deepl_client = deepl.DeepLClient(DEEPL_API) #to move below? ## yes

    trad_block = []

    for idx, box in enumerate(blocks):
        coord = box[0]
        text = box[1] #text
        n_lines = box[2] #integer

        trad = deepl_client.translate_text(text,
                                           source_lang=source_l,
                                           target_lang=target_l)
        

        # fake translation
        formatted_trad = split_text(trad,n_lines)
        
        trad_block.append((coord,text,formatted_trad))

    return trad_block


def apply_translation(image_bytes,translated_text):

    font_path = ffont_path("DejaVuSans.ttf")

    image = Image.open(BytesIO(image_bytes))
    draw = ImageDraw.Draw(image)
    font_txt = ImageFont.truetype(font_path, 13)
    for i in translated_text:

        coordinates = i[0]
        trad = i[2]

        draw.rectangle(coordinates, fill="white")
        draw.multiline_text(
            xy=coordinates[0],
            # anchor='mm',
            text=trad,
            fill ="black",
            font=font_txt ,
            align="center"
            )
        
    return image


def pdf_file(pages):
    images = [i.convert('RGB') for i in pages]

    pdf_bytes = BytesIO()
    images[0].save(pdf_bytes, format='PDF', save_all=True, append_images=images[1:])
    pdf_bytes.seek(0)

    return pdf_bytes


def ffont_path(preferred_font="DejaVuSans.ttf"):

    system = platform.system()

    if system == "Windows":
        font_dir = os.path.join(os.environ['WINDIR'], "Fonts")
        font_path = os.path.join(font_dir, preferred_font)

    elif system == "Linux":
        font_path = "/usr/share/fonts/truetype/dejavu"

    return font_path


def main():

    ## to be params in the future
    manga_lang = 'es'
    _url = "https://mangapark.net/title/301153-es_419-hadacamera/9280970-vol-6-ch-50"
    thres = 35 # pixel difference between mean x and y axis of previous text box to following text box to be considered as part of the same sentence.
    comic_name = "hadacamera"
    chapter_name = comic_name+'-'+_url[-7::]

    ## ==== Actual code starts here ====================================

    # if not os.path.isdir(f"{root_folder}/{comic_name}"):
    #     os.mkdir(f"{root_folder}/{comic_name}")
    root_folder = Path(__file__).parent / "output"
    comic_folder = root_folder / comic_name
    comic_folder.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(
        [manga_lang.lower(),'en'],
        detector='DB',
        gpu=torch.cuda.is_available()
    ) 
    
    chap_translated = []

    ## as you just iterate them, why not a generator ?
    img_list = get_images(_url)
    for idx, image in enumerate(img_list):

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

        chap_translated.append(img_trans)

        ## maybe logging ? maybe later
        print(f'Processed the {idx} image')


    with open(f"{root_folder}/{chapter_name}.pdf", "wb") as f:
        f.write(pdf_file(chap_translated).getvalue())


if __name__ == '__main__':
    main()


