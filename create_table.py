import pandas as pd
import countryflag

from daisy.abstract import LANGUAGES

data_dict = {
    "language_iso2": [],
    "language": [],
    "language_en": [],
    "sources": [],
    "countries": [],
    "date range": [],
}

country_map = {
    "UK": "United Kingdom",
    "Korea": "South Korea",
    "Macedonia": "North Macedonia",
    "España": "Spain",
    "UAE": "United Arab Emirates",
    "Palestinian Territories": "Palestine",
    "USA": "United States",
    "US": "United States",
}


def map_country(country):
    if country in country_map:
        return country_map[country]
    return country


additional_emojis = {
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
}

for language in LANGUAGES:
    df = pd.read_csv(f"data/{language}/sampled_items.csv")
    df["date"] = pd.to_datetime(df["date"])
    data_dict["language_iso2"].append(LANGUAGES[language].iso2)
    data_dict["language"].append(LANGUAGES[language].native_name)
    data_dict["language_en"].append(LANGUAGES[language].english_name)
    data_dict["sources"].append(", ".join(df["name"].value_counts().index[:3]) + "...")
    # map countries based on country_map for the whole dataframe
    df["country"] = df["country"].map(map_country)
    country_counts = []
    for country in df["country"].value_counts().index:
        _country = country
        try:
            emoji = countryflag.getflag(country)
        except:
            if country in additional_emojis:
                emoji = additional_emojis[country]
            else:
                emoji = None
        if emoji is not None:
            country_counts.append(
                f"{_country} {emoji} ({df['country'].value_counts()[_country]})"
            )
        else:
            country_counts.append(
                f"{_country} ({df['country'].value_counts()[_country]})"
            )
    data_dict["countries"].append(", ".join(country_counts))
    start_date_formatted = df["date"].min().strftime("%Y-%m-%d")
    end_date_formatted = df["date"].max().strftime("%Y-%m-%d")
    data_dict["date range"].append(f"{start_date_formatted} - {end_date_formatted}")

df = pd.DataFrame(data_dict)
# rename columns for markdown
df.columns = [
    "Language ISO 2",
    "Language",
    "Language (English)",
    "Sources",
    "Countries",
    "Date Range",
]
df.to_markdown("data_summary.md", index=False)
