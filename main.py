
# import chain: 
# interfaces.py 
# -> translator.py 
#     -> chapter.py (also depends on interfaces.py) 
#         -> main.py

from deepl import DeepLClient
from APIs import DEEPL_API

from scripts.chapter import process_chapter


def main():

    ## get inputs from somewhere
    source_lang = 'es'
    url = "https://mangapark.net/title/301153-es_419-hadacamera/9280970-vol-6-ch-50"
    comic_name = "hadacamera"
    chapter_name = comic_name+'-'+url[-7::]

    deepl_client = DeepLClient(DEEPL_API)

    process_chapter(
        language=source_lang,
        url=url,
        comic=comic_name,
        name=chapter_name,
        # the following 2 are kw only, but are explicitly optional
        reader=None,
        gpu = False,
        deepl_client=deepl_client
    )


if __name__ == "__main__":
    main()
