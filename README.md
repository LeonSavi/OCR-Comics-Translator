# OCR Comics Translator

A Python tool that extracts text from manga/comic images from a website using OCR and translates it into English or another target language using DeepL API. Designed to be lightweight, flexible, and customisable.

## Motivation

I started this project independently because I wanted to read a manga series that was partially translated into English, with the remaining chapters only available in Spanish. Waiting a year for an official release was off the table -- so I decided to build my own solution.

As most of the OCR chrome extensions did not work or had a limited amount of tokens daily, I decided to take this challenge and leverage my python experience to create something, easy to use and applicable in multiple context. Also with hope, to learn more about programming and the field of Data Science.

## Requirements

- python >= 3.10 or python < 3.12
- Dependencies listed in [`requirements.txt`](./requirements.txt)
- (Optional) WSL + Miniconda for Windows users
- (Optional) Java Runtime Environment (required for grammar correction via `language_tool_python`)

## Installation

0. OPTIONAL: for Windows user, I highly recommend setting up to ease the installation:
- [WSL (Windows Subsystem Linux)](https://youtu.be/gTf32sX9ci0?si=2s0JSHl26bxGqnjw)
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)

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

<pre> ```
ocr-comics-translator/
├── APIs.py
├── LICENSE
├── README.md
├── main.py
├── output
├── requirements.txt
├── scripts
│   ├── __init__.py
│   ├── chapter.py
│   ├── cleaner.py
│   ├── interfaces.py
│   └── translator.py
└── tree.txt
``` </pre>

## Simple Usage

Upon completing the installation, In [`main.py`](./main.py) change the **url** with the desired chapter for the website below, and change the source_lang and the name of the comic. Then you can run the script. You will find the .pdf of the chapter in the folder `output/{comic name}`.

Tested websites:
- [Mangapark](https://mangapark.io/)

## Scripts Explaination

- **main.py**:
- **scripts/chapter.py**:
- **scripts/translator.py**:
- **scripts/cleaner.py**:
- **scripts/interfaces.py**:

## Contributors
- LS: Owner/Maintener
- GB: Contributor