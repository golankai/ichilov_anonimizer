"""
A script to run the SafeHarbor tool on text data to anonymize it.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json  # noqa: E402
from collections import Counter  # noqa: E402
import pandas as pd  # noqa: E402
import re  # noqa: E402
import numpy as np  # noqa: E402
from hebsafeharbor import HebSafeHarbor  # noqa: E402

from transformers import set_seed  # noqa: E402

ALLOW_LIST = []

DR_PREFIX = r"(?:ד\"?ר\'?|פרופ\'?|פר\'?|פרופסור|דוקטור)"

PREFIXES_REG = r"(?:ו|מ|ל|ומ|ול|ש|וש|מש|וכש|ומש|ומכש|של|כשל|כש|כשמ|לכש)"

PHONE_COLUMNS = ["phone_number"] + [f"phone_number_{i}" for i in range(1, 4)]
NAMES_COLUMNS = [
    "first_name",
    "last_name",
    "first_name_eng",
    "last_name_eng",
    "father_name",
    "mother_name",
]
ANON_COLUMNS = (
    PHONE_COLUMNS
    + NAMES_COLUMNS
    + ["email", "PatID", "PatIdNumZ", "birth_date", "death_date", "zipcode", "address"]
)

PATNUM_NOT_FOUND = "PATNUM_NOT_FOUND"

# read the list of organizations
orgs_path = "safe_harbor/organizations.csv"
ORGS_LIST = pd.read_csv(orgs_path, encoding="utf-8-sig")["name"].tolist()

def setup_hsh():
    # Set up SafeHarbor
    safe_harbor = HebSafeHarbor()

    def anonymize(text):
        doc = {"text": text}
        output = safe_harbor([doc])[0]
        return output

    return anonymize


def get_anonymized_items(row):
    x = row["anon_output"]
    items, types, scores = [], [], []
    for item in x.consolidated_results:
        items.append(x.text[item.start : item.end])
        types.append(item.entity_type)
        scores.append(item.score)

    return pd.Series(
        [items, types, scores], index=["anonymized_items", "types", "scores"]
    )


def main_hsh(dataset, anonymized_items_file: str = None):
    anonymize = setup_hsh()

    # Anonymize data
    dataset["anon_output"] = dataset["text"].apply(anonymize)
    dataset["anonymized_text"] = dataset["anon_output"].apply(
        lambda x: x.anonymized_text.text.strip()
    )

    # Get anonymized items
    dataset[["anonymized_items", "types", "scores"]] = dataset.apply(
        lambda row: get_anonymized_items(row), axis=1
    )

    # Remove output column
    dataset = dataset.drop(columns=["anon_output"])

    if anonymized_items_file:
        # Get list of anonymized items
        anonymized_items = []
        dataset["anonymized_items"].apply(lambda x: anonymized_items.extend(x))
        anonymized_items = Counter(anonymized_items)
        anonymized_items = anonymized_items.most_common()
        anonymized_items = dict(anonymized_items)

        # Save anonymized items
        with open(anonymized_items_file, "w", encoding="utf-8") as f:
            json.dump(anonymized_items, f, ensure_ascii=False, indent=0)

    return dataset


def get_random_digits(n, begin_zero=True):
    if begin_zero:
        start_range = 0
    else:
        start_range = 1
    return "".join([str(np.random.randint(start_range, 10)) for _ in range(n)])


def get_random_day():
    return str(np.random.randint(1, 29))


def replace_with_random_day(match, year_first=False):
    random_day = get_random_day()
    if year_first:
        return rf"{match.group(1)}{random_day}"
    else:
        return rf"{match.group(1)}{random_day}{match.group(2)}"


def anon_names(text, demog_row, *args):
    names = list(demog_row[NAMES_COLUMNS].values[0])

    # Create a list of splitted names to handle second names and remove empty strings
    names = [name.strip() for to_split in names for name in to_split.split()]
    names = list(set([re.escape(name) for name in names if name and name != " "]))


    ben_bat_flag = None
    if "בת" in names:
        names.remove("בת")
        ben_bat_flag = "בת"
    if "בן" in names:
        names.remove("בן")
        ben_bat_flag = "בן"

    all_names = rf"(?:{'|'.join(names)})"
    names_regex = rf"\b{PREFIXES_REG}?{all_names}\b"
    found_names = re.findall(names_regex, text)
    anonymized_text = re.sub(names_regex, "<שם>", text, flags=re.IGNORECASE)

    # Anonymize ben/bat
    if ben_bat_flag:
        # Anonymize ben/bat only if NOT followed by a space and then a digit
        ben_bat_regex = rf"\b{ben_bat_flag}(?!\s\d)\b"
        found_names.extend(re.findall(ben_bat_regex, anonymized_text))
        anonymized_text = re.sub(
            ben_bat_regex, "<שם>", anonymized_text, flags=re.IGNORECASE
        )
    return anonymized_text, found_names


def anon_doctors(text, *args):
    anonymized_items = re.findall(rf"(\b{PREFIXES_REG}?{DR_PREFIX}\b\s\w+)", text)

    anonymized_text = re.sub(
        rf"(\b{PREFIXES_REG}?{DR_PREFIX}\b\s)\w+", r"\1<שם>", text, flags=re.IGNORECASE
    )
    return anonymized_text, anonymized_items


def get_random_phone(match=None):
    return f"05{get_random_digits(1)}-{get_random_digits(7)}"


def anon_phone_numbers(text, demog_row, *args): 
    anonymized_text = text

    # Anonymize random phone numbers
    phones_regex = r"(?:\+\d{2,3}|0)-?\d{1,3}-?-?\d{2,4}-?\d{0,4}"
    anonymized_items = re.findall(phones_regex, text)
    anonymized_items = list(filter(lambda x: len(x) > 9, anonymized_items))
    
    if anonymized_items:
        # Escape each phone number to safely construct the regex
        escaped_items = [re.escape(item) for item in anonymized_items]
        dynamic_regex = rf"(?:{'|'.join(escaped_items)})"
        anonymized_text = re.sub(
            dynamic_regex,
            get_random_phone,  # Callable function for dynamic replacement
            text,
            flags=re.IGNORECASE,
        )

    # Anonymize given phones if somehow they got away
    phones = list(demog_row[PHONE_COLUMNS].values[0])
    phones = [re.escape(str(phone)) for phone in phones if phone]
    if phones:
        all_phones = rf"(?:{'|'.join(phones)})"
        phones_regex = rf"\b{all_phones}\b"
        
        anonymized_items.extend(re.findall(phones_regex, anonymized_text))
        anonymized_text = re.sub(
            phones_regex,
            get_random_phone,  # Callable function for dynamic replacement
            anonymized_text,
            flags=re.IGNORECASE,
        )

    return anonymized_text, anonymized_items


def get_random_dr_number(match):
    return rf"{match.group(1)}{get_random_digits(5)}"


def anon_dr_numbers(text, *args):
    DR_NR_PREFIX = r"(?:מ\.?ר\.?|מספר רופא)"
    anonymized_items = re.findall(rf"{DR_NR_PREFIX}\s?\d{{1,5}}", text)
    anonymized_text = re.sub(
        rf"({DR_NR_PREFIX}\s?)\d{{1,5}}",
        get_random_dr_number,
        text,
        flags=re.IGNORECASE,
    )
    return anonymized_text, anonymized_items


def anon_emails(text, demog_row, *args):
    anonymized_items = []
    anonymized_text = text
    email = demog_row["email"].values[0]
    email = re.escape(str((email)))
    if email and email != "NULL":
        email_regex = rf"\b{email}\b"
        anonymized_items = re.findall(email_regex, text)
        anonymized_text = re.sub(email_regex, "<מייל>", text, flags=re.IGNORECASE)

    # Anonymize random emails
    emails_regex = r"\b[\w\.\-_]+@[\w\.\-_]+\.\w+\b"
    anonymized_items.extend(re.findall(emails_regex, anonymized_text))
    anonymized_text = re.sub(
        emails_regex, "<מייל>", anonymized_text, flags=re.IGNORECASE
    )

    return anonymized_text, anonymized_items


def anon_orgs(text, *args):
    orgs = [org.strip() for org in ORGS_LIST]
    orgs = list(set([re.escape(org) for org in orgs]))
    # all_orgs = rf"(?:{'|'.join(orgs)}){{e<=2}}"
    all_orgs = rf"(?:{'|'.join(orgs)})"
    orgs_regex = rf"\b{PREFIXES_REG}?{all_orgs}\b"
    found_orgs = re.findall(orgs_regex, text)
    anonymized_text = re.sub(orgs_regex, "<ארגון>", text, flags=re.IGNORECASE)
    return anonymized_text, found_orgs


def anon_ids(text, demog_row, row):
    anonymized_items = []
    anonymized_text = text
    id = demog_row["PatID"].values[0]
    id = re.escape(str(id))
    if id is None or id == "NULL":
        id = row["PatIdNumZ"]
    if id and id != "NULL":
        anonymized_items = re.findall(rf"\b{id}\b", text)
        anonymized_text = re.sub(
            rf"\b{id}\b",
            lambda match: get_random_digits(9, begin_zero=False),
            text,
            flags=re.IGNORECASE,
        )

    # Anonymize random ids
    ids_regex = r"\b[^0]\d{8}\b"
    anonymized_items.extend(re.findall(ids_regex, anonymized_text))
    anonymized_text = re.sub(
        ids_regex,
        lambda match: get_random_digits(9, begin_zero=False),
        anonymized_text,
        flags=re.IGNORECASE,
    )
    return anonymized_text, anonymized_items


def anon_dates(text, *args):
    # Anonymize all dates
    anonymized_items = re.findall(r"\d{1,4}[\\/.]\d{1,2}[\\/.]\d{1,4}", text)
    # Anonymize dates, the days appear first
    anonymized_text = re.sub(
        r"([^\d])\d{1,2}([\\/.]\d{1,2}[\\/.]\d{1,4})",
        replace_with_random_day,
        text,
        flags=re.IGNORECASE,
    )
    # Anonymize dates, the years appear first
    anonymized_text = re.sub(
        r"(\d{4}[\\/.]\d{1,2}[\\/.])\d{1,2}",
        lambda match: replace_with_random_day(match, year_first=True),
        anonymized_text,
        flags=re.IGNORECASE,
    )
    return anonymized_text, anonymized_items


def anon_zip_codes(text, demog_row, *args):
    anonymized_items = []
    anonymized_text = text
    zip = demog_row["zipcode"].values[0]
    zip = re.escape(zip)
    if zip:
        anonymized_items = re.findall(rf"\b{zip}\b", text)
        anonymized_text = re.sub(
            rf"\b{zip}\b", lambda match: get_random_digits(7), text, flags=re.IGNORECASE
        )

    # Anonymize random zip codes
    zip_regex = r"[א-ת](\b\d{7}\b)"
    anonymized_items.extend(re.findall(zip_regex, text))
    anonymized_text = re.sub(
        zip_regex, lambda match: get_random_digits(7), text, flags=re.IGNORECASE
    )

    return anonymized_text, anonymized_items


def anon_addresses(text, demog_row, *args):
    address = demog_row["address"].values[0]
    address = re.escape(address)
    if address:
        anonymized_items = re.findall(rf"\b{address}\b", text)
        anonymized_text = re.sub(
            rf"\b{address}\b", "<כתובת>", text, flags=re.IGNORECASE
        )
        return anonymized_text, anonymized_items
    else:
        return text, []


def main_regex(dataset, demog_df):
    anon_functions = [
        anon_names,
        anon_doctors,
        anon_phone_numbers,
        anon_dr_numbers,
        anon_emails,
        anon_ids,
        anon_dates,
        anon_zip_codes,
        anon_addresses,
        anon_orgs,
    ]

    def anonymize(row):
        # Get patient number and text
        pat_num = row["PatNum"]
        anonymized_items = []
        text = row["text"]

        # Check if the patient number is in the demographic data
        if pat_num in demog_df["PatNum"].values:
            demog_row = demog_df[demog_df["PatNum"] == pat_num]
        # Remove 3 leading zeros from the patient number and check again
        elif str(pat_num)[3:] in demog_df["PatNum"].values:
            pat_num = str(pat_num)[3:]
            demog_row = demog_df[demog_df["PatNum"] == pat_num]
        else:
            # Return a marker for rows that won't be anonymized
            print(f"PATNUM not found:{pat_num}")
            return pd.Series(["", [PATNUM_NOT_FOUND]])

        anonymized_text = text
        for anon_func in anon_functions:
            anonymized_text, cur_items = anon_func(anonymized_text, demog_row, row)
            anonymized_items.extend(cur_items)

        return pd.Series([anonymized_text, anonymized_items])

    # Apply the anonymize function
    dataset[["anonymized_text", "anonymized_items"]] = dataset.apply(
        lambda row: anonymize(row), axis=1
    )

    # Count and exclude rows with PATNUM_NOT_FOUND
    not_found_count = (dataset["anonymized_items"].apply(lambda x: PATNUM_NOT_FOUND in x)).sum()
    print(f"Number of rows with PATNUM_NOT_FOUND: {not_found_count}")

    # Filter out rows with PATNUM_NOT_FOUND
    dataset = dataset[~dataset["anonymized_items"].apply(lambda x: PATNUM_NOT_FOUND in x)]
    print(f"Dataset length: {len(dataset)}")

    # Return the filtered dataset
    return dataset



def main_joint(dataset: pd.DataFrame, demog_df: pd.DataFrame) -> pd.DataFrame:
    def combine_anonymized_texts(row: pd.Series):
        """
        Combine anonymized texts from different methods.
        Use the regex method as a base and add some parts of the hsh method.
        :param row: A row in the dataset.
        """
        anonymized_text = row["anonymized_regex"]
        anonymized_items = row["items_regex"]
        hsh_items = row["items_hsh"]
        hsh_types = row["types"]
        hsh_scores = row["scores"]

        # Add some parts of the hsh method
        for item, ent_type, score in zip(hsh_items, hsh_types, hsh_scores):
            if ent_type in ["ORG"] and score > 0.7:
                anonymized_text = anonymized_text.replace(item, "<ארגון>")
                anonymized_items.append(item)
            elif ent_type in ["LOC"] and score > 0.8:
                anonymized_text = anonymized_text.replace(item, "<מיקום>")
                anonymized_items.append(item)
            else:
                continue

        return pd.Series([anonymized_text, anonymized_items])

    dataset = main_hsh(dataset)
    # Change names of new columns
    dataset = dataset.rename(
        columns={
            "anonymized_text": "anonymized_hsh",
            "anonymized_items": "items_hsh",
        }
    )
    dataset = main_regex(dataset, demog_df)
    # Change names of new columns
    dataset = dataset.rename(
        columns={
            "anonymized_text": "anonymized_regex",
            "anonymized_items": "items_regex",
        }
    )

    # Combine anonymized texts
    dataset[["anonymized_text", "anonymized_items"]] = dataset.apply(
        lambda row: combine_anonymized_texts(row), axis=1
    )

    return dataset


if __name__ == "__main__":
    # Set configurations
    METHOD = "regex"  # hsh or regex or joint
    DEBUG = False
    SEED = 42
    data_dir = "further_pre_training/data/cut_ichilov/"
    data_file = "ichilov_train_cut.parquet"
    # data_file = "ichilov_test_10k_cut.parquet"

    demog_file = "demog.parquet"

    # Set paths
    data_path = data_dir + data_file
    demog_path = data_dir + demog_file
    output_file = (
        data_dir
        + data_file.split(".")[0]
        + f"_anonymized_{METHOD}."
        + data_file.split(".")[1]
    )
    anonymized_items_file = (
        data_dir + data_file.split(".")[0] + f"_anonymized_items_{METHOD}.json"
    )

    # Set up seed
    set_seed(SEED)

    # Load data
    extension = data_file.split(".")[-1]
    if extension == "csv":
        dataset = pd.read_csv(data_path, encoding="utf-8-sig")
    elif extension == "parquet":
        dataset = pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported file type")

    # Sample data for debugging
    if DEBUG:
        dataset = dataset.sample(10)

    if demog_file and METHOD != "hsh":
        extension = demog_file.split(".")[-1]
        if extension == "csv":
            demog_df = pd.read_csv(demog_path, encoding="utf-8-sig")
        elif extension == "parquet":
            demog_df = pd.read_parquet(demog_path)
        else:
            raise ValueError("Unsupported file type")

        # Strip names
        demog_df[ANON_COLUMNS] = demog_df[ANON_COLUMNS].astype(str).apply(lambda x: x.str.strip())
        # Delete all "NULL" values
        demog_df = demog_df.replace("NULL", "")

    if METHOD == "hsh":
        dataset = main_hsh(dataset, anonymized_items_file)
    elif METHOD == "regex":
        dataset = main_regex(dataset, demog_df)
    elif METHOD == "joint":
        dataset = main_joint(dataset, demog_df)
    else:
        raise ValueError("Unsupported anonymization method")

    # Save anonymized data
    if extension == "csv":
        dataset.to_csv(output_file, index=False, encoding="utf-8-sig")
    elif extension == "parquet":
        dataset.to_parquet(output_file)
    else:
        raise ValueError("Unsupported file type")
