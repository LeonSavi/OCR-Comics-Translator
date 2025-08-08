# OCR Comics Translator

A Python tool that extracts text from manga/comic images from a website using OCR and translates it into English or another target language using DeepL API.

## Motivation

I started this project independently because I wanted to read a manga series that was partially translated into English, with the remaining chapters only available in Spanish. Waiting a year for an official release was off the table - so I decided to build my own solution.

As most of the OCR chrome extensions did not work or had a limited amount of tokens, I decided to take this challenge and leverage my python experience to create something that easy to use and applicable in multiple context. 

This little experince was intellectually stimulating, as I choose to **not** rely on quick, but average/mediocre, Genererative AI solutions but to seek the coding answers by my own. Although I recognise that AI is amazing for productivity at workplace, to rely excessively on it lead to mental lazyness - especially for analysts that have little prior programming experience. The long-forgotten Stackoverflow, as well as Youtube, and Books, have proven to be far more valuable to develop innovative solutions, learn how things work under the hood and finding answers that truly sticks.

## Requirements

- python = 3.10 | python = 3.11
- Dependencies listed in [`requirements.txt`](./requirements.txt)
- (Optional) [WSL (Windows Subsystem Linux)](https://youtu.be/gTf32sX9ci0?si=2s0JSHl26bxGqnjw) + [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer) - Recommended for Windows users
- (Optional) Java Runtime Environment (required for grammar correction via `language_tool_python`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/ocr-comics-translator.git
cd ocr-comics-translator
```

2. Create the Python Env and install the dependencies:
```bash
conda create -n OCRcomics python=3.10
conda activate OCRcomics

pip install -r requirements.txt
```

3. Get an [DeepL API](https://developers.deepl.com/docs/getting-started/intro):
```bash
cd ocr-comics-translator
touch APIs.py
echo 'DEEPL_API = " deepL_api_key_here "' > APIs.py
```

4. OPTIONAL: Install Java for the grammar checker:
```bash
sudo apt update
sudo apt install default-jre -y
```

## File Structure

<pre> 
ocr-comics-translator/
├── APIs.py
├── LICENSE
├── README.md
├── main.py
├── output
├── requirements.txt
└── scripts
   ├── __init__.py
   ├── chapter.py
   ├── cleaner.py
   ├── interfaces.py
   └── translator.py
 </pre>

## Simple Usage

Upon completing the installation, In [`main.py`](./main.py) change the **url** with the desired chapter for one of the listed website below, and change the `source_lang` and the name of the comic. Then you can run the script. You will find the .pdf of the chapter in the folder `output/{comic name}`.

### Tested Websites:
- [Mangapark](https://mangapark.io/)

## Scripts Explaination

- **main.py**: Use this script to process and translate an entire manga chapter.
- **scripts/chapter.py**: Handles Chapter-level logic by wrapping the url, source language, etc into a `Chapter` object, it retrieves the images apply the translation pipeline page by page.
- **scripts/translator.py**: Pages are translated individually via `PageTranslator` class. Text is detected using the library `easyocr`, merged to identify text boxes, cleaned, translates it with DeepL API, and pastes the translated text back onto the image.
- **scripts/cleaner.py**: Contains the `TextCleaner` class, which performs grammar correction and light polishing using `language_tool_python`. Japanese is not fully supported.
- **scripts/interfaces.py**: Defines the core data structures:
  -  `Point`,`Box` for Geometry
  -  `ReadText`,`FoundText`,`FormattedText` to deal with `easyocr` outputs.

## Contributors
- [@LeonSavi](https://github.com/LeonSavi) - Creator & Maintainer  
- [@GBeggiato](https://github.com/GBeggiato) - Contributor
