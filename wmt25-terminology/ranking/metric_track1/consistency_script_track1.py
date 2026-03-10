import pandas as pd
import os
import json
import time

import nltk
nltk.download('punkt_tab')

from termbasedmetric import TermBasedMetric

import argparse

def run_cycle(src_lang, tgt_lang, mode, filepath):
    '''
    runs evaluation for the 1st track, for a given source, target language and mode,
        collecting the statistics and the extracted data (terms, alignments, pseudoreferences)

    :param src_lang: source language
    :param tgt_lang: target language
    :param mode: evaluation mode ('proper', 'random', 'noterm')
    :param filepath: path to the directory containing the submitted files
    '''

    # collect all statistics in the format: {system: {pseudoref_choice: {macro: score, micro: score}}}
    total_stats = {}
    # read all submitted files
    print(os.listdir(filepath))
    file_list = []
    for dir1 in os.listdir(filepath):
        dir1_path = os.path.join(filepath, dir1)
        if os.path.isdir(dir1_path):
            for f in os.listdir(dir1_path):
                file_list.append(os.path.join(dir1, f))
    
    print("file_list: ", file_list)
    # file_list = [f for f in os.listdir(filepath) if f.endswith('.jsonl') and src_lang+tgt_lang in f and mode in f]

    # initializing term based metric
    TBM = TermBasedMetric(src_lang, tgt_lang, 'predefined', 'llm')

    # catching files with raised errors (deprecated)
    error_files = []
    print("file_list: ", file_list)
    for file in file_list:
        print(file)
        
        system_name = file.split('.')[0] #.split('/')[-1]
        # loading the submitted file
        TBM.load(system_name, mode=mode, file_path='../submissions/')
        # extracting keywords
        TBM.extract_keywords()
        # aligning source terms to translated sentences
        print("aligning with qwen2.5-0.5b-instruct: ", system_name)
        TBM.align(test=True)
        pseudoref_choices = ['first', 'frequent', 'predefined']
        var_dict = {pseudoref_choice: {'micro': 0, 'macro': 0} for pseudoref_choice in pseudoref_choices}
        flat_df = []
        for pseudoref_choice in pseudoref_choices:
            # assigning pseudoreferences for all three pseudoreference types (first, frequent, predefined)
            _, _, _, sel = TBM.assign_pseudoreferences(pseudoref_choice)
            # computing the micro and macro averaged accuracies for pseudoreference and aligned term overlaps
            micro, flat_micro = TBM.compute_metric('micro')
            macro, _ = TBM.compute_metric('macro')
            if pseudoref_choice == 'first':
                sel_first = flat_micro.copy(deep=True)
            elif pseudoref_choice == 'frequent':
                sel_freq = flat_micro.copy(deep=True)
            flat_micro.rename(columns={'pseudoref': f'pseudoref_{pseudoref_choice}'}, inplace=True)
            flat_df.append(flat_micro)
            var_dict[pseudoref_choice]['micro'] = micro
            var_dict[pseudoref_choice]['macro'] = macro
        # adding all statistics to total dict
        total_stats[system_name] = var_dict
        flat_df = pd.concat(flat_df, axis=1)
        # saving the file with pseudoreferences (and aligned term translations)
        path = f'pseudorefs/{system_name}_pseudoref.tsv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        flat_df.to_csv(path, sep='\t')
        # saving the file with all initial data (sentences, terms, extracted translations)
        path = f'processed/{system_name}_processed.tsv' 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        TBM.bitext_df[['terms', 'src_raw', 'mt_raw', 'src_terms', 'alg_terms', 'norm_tgt_terms']].to_csv(path, sep='\t')

    return total_stats, error_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--srclang')
    parser.add_argument('-t', '--tgtlang')
    parser.add_argument('-m', '--mode')
#    parser.add_argument('-l', '--track')
    args=parser.parse_args()
    print(args.srclang, args.tgtlang, args.mode)
    a = time.time()
    # stats_dict, error_files = run_cycle(args.srclang, args.tgtlang, args.mode, f'data/submissions/track1')
    
    # modified for outputs
    stats_dict, error_files = run_cycle(args.srclang, args.tgtlang, args.mode, f'../submissions/track1')

    with open("scores/track1_score_dict.json") as f:
        score_dict = json.load(f)

        mapping_dict = {
            "frequent": "consistency_frequent",
            "predefined": "consistency_predefined"
        }

        for team in stats_dict:
            for pseudoref_choice in mapping_dict:
                score_dict[args.tgtlang][args.mode][team][mapping_dict[pseudoref_choice]]['macro'] = stats_dict[f"{team}/{team}"][pseudoref_choice]['macro']
                # score_dict[args.tgtlang][args.mode][team][mapping_dict[pseudoref_choice]]['macro'] = stats_dict[team][pseudoref_choice]['micro']

        json.dump(score_dict, open("scores/track1_score_dict_full.json", "w"))

    b = time.time()
    print(f'finished in {b-a} seconds')
    print(error_files)
    with open(f'stats/stats-{args.srclang}{args.tgtlang}.{args.mode}.json', 'w') as fp:
        json.dump(stats_dict, fp)


    # we want [teamname][frequent/predefined][macro]