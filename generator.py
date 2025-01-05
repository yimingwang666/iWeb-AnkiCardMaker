from mdict_query import IndexBuilder
import os
import logging
from lxml import html
import cloudscraper
import requests
import re
import pandas as pd
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# Define color codes for logging
class LogColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m'  # Purple
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


# Configure logging
log_formatter = LogColorFormatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler = logging.StreamHandler()
log_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[log_handler])

# Initialize dictionary indexes
mdx_tld = IndexBuilder('mdx/TLD.mdx')
mdx_oaldpe = IndexBuilder('mdx/oaldpe.mdx')
mdx_ode = IndexBuilder('mdx/ODE Living Online.mdx')


def parse_tld(word: str):
    """Parse the TLD dictionary for the given word."""
    try:
        entry = mdx_tld.mdx_lookup(word)

        if entry:
            tree = html.fromstring(entry[0])

            # Extract part of speech and frequency
            pos = tree.xpath('//div[@class="coca iweb"]/span[@class="pos"]/text()')
            pos_total = tree.xpath('//div[@class="coca iweb"]/div[@class="total"]/text()')
            pos_total = [int(total) for total in pos_total]
            pos_freq = [f'{(round(total / sum(pos_total) * 100, 1))}%' for total in pos_total]

            # Extract Chinese definitions and frequencies
            cn_def = tree.xpath('//div[@class="coca2"]/text()')
            cn_def = [cn.replace("(", "").replace(")", "").replace(",", "") for cn in cn_def][:-1]
            def_freq = tree.xpath('//div[@class="coca2"]/font/text()')

            # Extract Chinese definitions with part of speech
            def_pos = tree.xpath('//div[@class="dcb"]/span[@class="pos"]/text()')
            def_cn = tree.xpath('//div[@class="dcb"]/span[@class="dcn"]/text()')

            return pos, pos_freq, cn_def, def_freq, def_pos, def_cn

        logging.warning(f"TLD entry not found: {word}")
        return None, None, None, None, None, None

    except Exception as e:
        logging.error(f"TLD parsing failed: {word}, error: {e}")
        return None, None, None, None, None, None


def parse_oaldpe(word: str):
    """Parse the OALDPE dictionary for the given word."""
    try:
        entries = mdx_oaldpe.mdx_lookup(word.lower())

        for entry in entries:
            tree = html.fromstring(entry)
            if tree.xpath('//h1/text()') and word == tree.xpath('//h1/text()')[0]:
                tree = html.fromstring(entry)

                try:
                    # Extract British and American phonetics
                    phons_br = tree.xpath('//div[@class="phons_br"]/span[@class="phon"]/text()')[0]
                    phons_am = tree.xpath('//div[@class="phons_n_am"]/span[@class="phon"]/text()')[0]

                    return phons_br, phons_am
                except Exception as e:
                    logging.error(f"OALDPE Phons parsing failed: {word}, error: {e}")
                    return None, None
        else:
            logging.warning(f"OALDPE entry not found: {word}")
            return None, None

    except Exception as e:
        logging.error(f"OALDPE parsing failed: {word}, error: {e}")
        return None, None


def parse_ode(word: str):
    """Parse the ODE dictionary for the given word."""
    try:
        entry = mdx_ode.mdx_lookup(word)

        if entry:
            tree = html.fromstring(entry[0])

            try:
                # Extract British and American phonetics
                phons_br = tree.xpath('.//a[@class="phoneticSymbol"]/text()')[0]
                phons_br = f'/{phons_br}/'
                phons_am = tree.xpath('.//a[@class="phoneticSymbol us"]/text()')[0]
                phons_am = f'/{phons_am}/'

                return phons_br, phons_am
            except Exception as e:
                logging.error(f"ODE Phons parsing failed: {word}, error: {e}")
                return None, None

        logging.warning(f"ODE entry not found: {word}")
        return None, None

    except Exception as e:
        logging.error(f"ODE parsing failed: {word}, error: {e}")
        return None, None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
)
@sleep_and_retry
@limits(calls=5, period=1)
def get_vocabulary(word: str):
    """Fetch the definition of the word from Vocabulary.com using cloudscraper."""
    try:
        url = f"https://www.vocabulary.com/dictionary/{word}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Referer": "https://www.vocabulary.com/",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
        }

        scraper = cloudscraper.create_scraper()  # Initialize Cloudflare bypass
        response = scraper.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            vocabulary_def_en = re.findall(r'<p class="short">(.+?)</p>', response.text)
            if vocabulary_def_en:
                return vocabulary_def_en[0].replace('<i>', '').replace('</i>', '')
            else:
                logging.warning(f"Vocabulary definition not found: {word}")
                return None
        else:
            logging.error(f"Vocabulary request failed, status code: {response.status_code}, word: {word}")
            return None

    except Exception as e:
        logging.error(f"Vocabulary fetch failed: {word}, error: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
)
@sleep_and_retry
@limits(calls=2, period=1)
def get_tio(word: str, cutoff=2):
    """Fetch examples (length > cutoff) of the word from tio.freemdict.com."""
    try:
        url = "https://tio.freemdict.com/api"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        }
        params = {
            'key': word
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            example = []
            tree = html.fromstring(response.text)

            # Find all elements in the document
            elements = tree.xpath('//*')

            # Find the index of <hr />
            hr_index = next((i for i, elem in enumerate(elements) if elem.tag == 'hr'), -1)

            # Get all <span class="pg_exam"> after <hr />
            if hr_index != -1:
                pg_exam_elements = [elem for elem in elements[hr_index + 1:] if
                                    elem.tag == 'span' and elem.get('class') == 'pg_exam']
            else:
                pg_exam_elements = tree.xpath('//span[@class="pg_exam"]')

            # Extract and print English and Chinese texts
            for pg_exam in pg_exam_elements:
                # Extract English part
                english_texts = pg_exam.xpath('.//text()[not(ancestor::zh_cn)]')
                en = ''.join(english_texts).strip()

                if len(en.split()) <= cutoff:
                    continue

                # Extract Chinese part
                chinese_texts = pg_exam.xpath('.//zh_cn/text()[not(parent::span)]')
                cn = ''.join(chinese_texts).strip()

                example.append({'en': en, 'cn': cn})

            return example

        else:
            logging.error(f"Tio request failed, status code: {response.status_code}, word: {word}")
            return None

    except Exception as e:
        logging.error(f"TIO fetch failed: {word}, error: {e}")
        raise


def process_word(word: str):
    """Process a single word by fetching data from TLD, OALDPE, and Vocabulary.com."""
    logging.info(f"Processing word: {word}")
    try:
        pos, pos_freq, cn_def, def_freq, def_pos, def_cn = parse_tld(word)
        phons_br, phons_am = parse_oaldpe(word)
        vocabulary_def_en = get_vocabulary(word)
        example = get_tio(word)

        # ODE as a replacement of phonetic symbols source
        if not phons_br or not phons_am:
            phons_br, phons_am = parse_ode(word)

        return [word, phons_br, phons_am, vocabulary_def_en, pos, pos_freq, cn_def, def_freq, def_pos, def_cn, example]

    except Exception as e:
        logging.error(f"Word processing failed: {word}, error: {e}")
        return [word, None, None, None, None, None, None, None, None, None, None]


def load_processed_words(output_file):
    """Load already processed words from the output file."""
    if not os.path.exists(output_file):
        return set()

    df = pd.read_csv(output_file)
    return set(df['Word'])


def save_to_csv(output_file, results):
    """Save the results to a CSV file."""
    df = pd.DataFrame(results,
                      columns=["Word", "Phons_Br", "Phons_Am", "Def_En", "POS", "POS_Freq", "Cn_Def", "Def_Freq",
                               "Def_POS", "Def_Cn", "Examples"])
    if os.path.exists(output_file):
        df.to_csv(output_file, mode="a", index=False, header=False, encoding="utf-8")
    else:
        df.to_csv(output_file, index=False, encoding="utf-8")


def main(input_file, output_file, batch_size=100, max_workers=8):
    """Main function to process words in parallel and save results to a CSV file."""
    with open(input_file, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f]

    num = 0
    processed_words = load_processed_words(output_file)
    words_to_process = [word for word in words if word not in processed_words]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_word, word): word for word in words_to_process}

        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if len(results) >= batch_size:
                save_to_csv(output_file, results)
                num += 1
                logging.info(f"""
                -----------------
                {num * batch_size} WORDS SAVED
                -----------------
                """)
                results.clear()

        # Save any remaining results
        if results:
            save_to_csv(output_file, results)
            logging.info('Remaining words saved')


if __name__ == "__main__":
    name = 'iweb'
    input_file = f"vocab/{name}.txt"  # Input file path
    output_file = f"vocab/{name}.csv"  # Output file path

    main(input_file, output_file, max_workers=4)
