import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split


def train_test_split_dataset(
    df: pd.DataFrame, testing_samples, val_size=0.1, random_state=42, tag="image"
):
    """TRAIN TEST SPLIT: \n
    Splits a DataFrame or annotations file into training val and testing sets in a stratified way. \n
    Args:
        df (pd.DataFrame): DataFrame to split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
    Returns:
        Returns:
        pd.DataFrame: Training set.
        pd.DataFrame: Validation set.
        pd.DataFrame: Testing set.
    """

    # At first we will keep certain audio files separate for testing

    origin_sample = df["original_sample"].unique()

    original_test_samples = [i + "-original" for i in testing_samples]

    potential_fake_test_samples = [i + "-to" for i in testing_samples]

    # i want to keep the original samples in the test set that contain the strings from the potential_fake_test_samples

    fake_test_samples = [
        i for i in origin_sample if any(j in i for j in potential_fake_test_samples)
    ]

    test_df = df[df["original_sample"].isin(original_test_samples + fake_test_samples)]

    # Remove the test samples from the dataset

    df = df[~df["original_sample"].isin(original_test_samples + fake_test_samples)]

    # We will then split the remaining data into training and validation sets

    if tag == "image":

        # split off the train and val set

        train_df, val_df = train_test_split(
            df, test_size=val_size, stratify=df["LABEL"], random_state=random_state
        )

    elif tag == "tabular":

        scaler = StandardScaler()

        df.iloc[:, :-2] = scaler.fit_transform(df.iloc[:, :-2])

        # split off the train and val set

        train_df, val_df = train_test_split(
            df, test_size=val_size, stratify=df["LABEL"], random_state=random_state
        )

        # Before further splitting the data, normalize the features

        test_df.iloc[:, :-2] = scaler.transform(test_df.iloc[:, :-2])

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
