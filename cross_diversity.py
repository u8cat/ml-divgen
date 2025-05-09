import os, json, random
from typing import Iterable

import numpy as np
from functools import partial
from multiprocessing.pool import Pool
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from tqdm import tqdm

model_list = [
    'EleutherAI/pythia-2.8b',
    'ContextualAI/archangel_sft_pythia2-8b',
    'ContextualAI/archangel_sft-ppo_pythia2-8b',
    'ContextualAI/archangel_sft-dpo_pythia2-8b',
]

result_dir = 'model_outputs'
dataset_name = 'euclaise/writingprompts'
dataset_split = 'validation'
num_samples = 1000
n_seq = 10


def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return np.array(sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function))


def self_bleu(generations: Iterable[list[str]], n_sample=1000):
    random.seed(0)
    
    # Sample prompts if there are more than n_sample
    prompt_indices = list(range(len(generations)))
    if n_sample < len(prompt_indices):
        prompt_indices = random.sample(prompt_indices, n_sample)
    
    all_bleus = []
    smoothing_function = SmoothingFunction().method1
    weights = [
        (1.0, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (1.0 / 3, 1.0 / 3, 1.0 / 3, 0),
        (0.25, 0.25, 0.25, 0.25),
        (0.2, 0.2, 0.2, 0.2, 0.2)]
    
    # Process each prompt separately
    for idx in tqdm(prompt_indices, desc="Computing Self-BLEU"):
        prompt_sentences = [sentence.split() for sentence in generations[idx]]
            
        # Calculate self-BLEU for each generation in the prompt
        pool = Pool(processes=os.cpu_count())
        prompt_bleus = sum(
            pool.imap_unordered(
                partial(bleu_i, weights, prompt_sentences, smoothing_function),
                range(len(prompt_sentences))
            )
        ) / len(prompt_sentences)
        pool.close()
        pool.join()
        
        all_bleus.append(prompt_bleus)
    
    return np.mean(all_bleus, axis=0)


if __name__ == '__main__':
    for model_name_or_path in model_list:
        print(f'Metrics for {model_name_or_path}')
        result_file = f"{result_dir}/{model_name_or_path.replace('/', '_')}_{dataset_name.replace('/', '_')}_{dataset_split}_generations_{num_samples}_nseq{n_seq}.json"
        with open(result_file) as f:
            results: list[dict[str, str]] = json.load(f)
        assert len(results) == num_samples

        generations = list(map(lambda r: r['generated'], results))
        bleu1, bleu2, bleu3, bleu4, bleu5 = self_bleu(generations)
        print(f'bleu1 = {bleu1:.4f}')
        print(f'bleu2 = {bleu2:.4f}')
        print(f'bleu3 = {bleu3:.4f}')
        print(f'bleu4 = {bleu4:.4f}')
        print(f'bleu5 = {bleu5:.4f}')

