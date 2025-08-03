
# import chain: 
# interfaces.py 
# -> translator.py 
#     -> chapter.py (also depends on interfaces.py) 
#         -> main.py

from deepl import DeepLClient
from deepL_api import api as DEEPL_API

from scripts.chapter import process_chapter


def main():

    ## get inputs from somewhere
    manga_lang = 'es'
    url = "https://mangapark.net/title/301153-es_419-hadacamera/9280970-vol-6-ch-50"
    comic_name = "hadacamera"
    chapter_name = comic_name+'-'+url[-7::]

    ## use them

    deepl_client = DeepLClient(DEEPL_API)

    process_chapter(
        language=manga_lang,
        url=url,
        comic=comic_name,
        name=chapter_name,
        # the following 2 are kw only, but are explicitly optional
        reader=None,
        deepl_client=deepl_client
    )


if __name__ == "__main__":
    main()
