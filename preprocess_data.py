import pyarrow
from datasets import load_dataset, Dataset

from tokenizer import Tokenizer

TITLE_TOKENS = "title_tokens"
ARTICLE_TOKENS = "article_tokens"


def main():
    ds = load_dataset('csv', data_files={
        "train": "data.csv"
    }, delimiter=',')

    training_set = ds["train"]

    tokenizer = Tokenizer()

    training_data: pyarrow.Table = training_set.data
    df = training_data.to_pandas()

    cutoff_length = 50

    df[TITLE_TOKENS] = df[:cutoff_length]['title'].apply(lambda x: tokenizer.tokenize(x, with_sod=False))
    df[ARTICLE_TOKENS] = df[:cutoff_length]['title'].apply(lambda x: tokenizer.tokenize(x, with_sod=False))

    df_trimmed = df.drop(columns=['id', 'url', 'text'])

    projected_training_data = pyarrow.Table.from_pandas(df_trimmed)

    hgf_Dataset = Dataset(projected_training_data[:cutoff_length])
    hgf_Dataset.to_json('tokens.json', index=False)

    # Test if Dataset JSON can be loaded.
    load_dataset('json', data_files={
        "train": "tokens.json"
    })

    tokenizer.save()

    print("Completed processing...")


if __name__ == "__main__":
    main()
