from collections import Counter
import torch
from operator import itemgetter

def get_text_features(text_list,model,tokenizer):
    with torch.no_grad():
        inputs = tokenizer(text_list,  padding=True, return_tensors="pt")
        text_features = model.get_text_features(**inputs)
    return text_features

def get_image_features(image_list,model,processor):
    with torch.no_grad():
        inputs = processor(images=image_list, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
    return image_features

def balance_on_characteristic(population_df,image_df,characteristic):
    characteristic_list = population_df[characteristic].tolist()
    sub_counts = Counter(characteristic_list)
    min_characteristic, min_count = min(sub_counts.items(), key=itemgetter(1))
    for c in sub_counts.keys():
        if c == min_characteristic:
            continue
        population_df = population_df.drop(population_df.query(f'{characteristic} == \'{c}\'').sample(sub_counts[c]-min_count).index)
    image_df = image_df.loc[population_df.index]
    return population_df, image_df

def balance_on_intersection(population_df,image_df,intersection):
    population_df['intersection'] = population_df[intersection].agg(' '.join, axis=1)
    intersection_counter = Counter(population_df['intersection'].tolist())
    min_characteristic, min_count = min(intersection_counter.items(), key=itemgetter(1))
    for c in intersection_counter.keys():
        if c == min_characteristic:
            continue
        population_df = population_df.drop(population_df.query(f'intersection == \'{c}\'').sample(intersection_counter[c]-min_count).index)
    image_df = image_df.loc[population_df.index]
    return population_df, image_df