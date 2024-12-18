# -*- coding: utf-8 -*-
"""LR_6_Task_2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hy3CXYRgE8KUMy20PJ1D2l31UcIRE5j9
"""

import pandas as pd
from itertools import product

data_url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
data = pd.read_csv(data_url)

columns_of_interest = ["price", "train_type", "origin", "destination", "train_class"]
data_cleaned = data[columns_of_interest].dropna()

price_labels = ["low", "medium", "high"]
data_cleaned["price_category"] = pd.cut(data_cleaned["price"], bins=3, labels=price_labels)

def count_occurrences(dataframe, group_by_cols):
    grouped = dataframe.groupby(group_by_cols).size()
    return grouped.to_dict()

train_type_frequencies = count_occurrences(data_cleaned, ["train_type", "price_category"])
origin_frequencies = count_occurrences(data_cleaned, ["origin", "price_category"])
destination_frequencies = count_occurrences(data_cleaned, ["destination", "price_category"])
class_frequencies = count_occurrences(data_cleaned, ["train_class", "price_category"])
price_category_counts = data_cleaned["price_category"].value_counts().to_dict()

def conditional_probability(category, feature_value, frequency_table, total_counts):
    if total_counts.get(category, 0) == 0:
        return 0
    return frequency_table.get((feature_value, category), 0) / total_counts[category]

selected_train_type = "AVE"
selected_origin = "SEVILLA"
selected_destination = "MADRID"
selected_class = "Preferente"

def calculate_probabilities(dataset, train_type, origin, destination, train_class):
    total_records = len(dataset)
    probabilities = {}

    for category in price_category_counts.keys():
        p_train_type = conditional_probability(
            category, train_type, train_type_frequencies, price_category_counts
        )
        p_origin = conditional_probability(
            category, origin, origin_frequencies, price_category_counts
        )
        p_destination = conditional_probability(
            category, destination, destination_frequencies, price_category_counts
        )
        p_class = conditional_probability(
            category, train_class, class_frequencies, price_category_counts
        )
        prior_prob = price_category_counts[category] / total_records

        probabilities[category] = p_train_type * p_origin * p_destination * p_class * prior_prob

    total_prob = sum(probabilities.values())
    if total_prob > 0:
        probabilities = {k: v / total_prob for k, v in probabilities.items()}

    return probabilities

resulting_probabilities = calculate_probabilities(
    data_cleaned, selected_train_type, selected_origin, selected_destination, selected_class
)

formatted_result = {cat: f"{prob:.2f}" for cat, prob in resulting_probabilities.items()}

print(
    f"Ймовірності для цінових категорій за параметрами "
    f"({selected_train_type}, {selected_class}, {selected_origin}, {selected_destination}):"
)
print(formatted_result)