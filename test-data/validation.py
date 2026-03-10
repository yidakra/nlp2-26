###
# VALIDATION SCRIPT FOR THE TERMINOLOGY SHARED TASK AT WMT2025
# INSTRUCTIONS:
# 0. HAVE PYTHON3 INSTALLED ON YOUR MACHINE; NO CUSTOM PACKAGES ARE REQUIRED
# 1. VALIDATE EACH TRACK (SENTENCE-LEVEL OR DOCUMENT-LEVEL MT) SEPARATELY
# 2. STORE ALL FILES OF YOUR SUMBISSION FOR A PARTICULAR TRACK IN A SINGLE (NON-NESTED) DIRECTORY, e.g. /your_submission/track1/submissions
# 3. STORE ALL INPUT FILES OF THIS TRACK IN ANOTHER (NON-NESTED) DIRECTORY, e.g. /your_submission/track1/inputs
# 4. RUN THE CODE WITH THE FOLLOWING PARAMETERS:
#       -t OR --track: CHOOSE 1 FOR SENTENCE-LEVEL OR 2 FOR DOCUMENT-LEVEL
#       -i OR --inputs: PATH TO YOUR FOLDER WITH INPUT FILES
#       -o OR --outputs: PATH TO THE FOLDER WITH YOUR SUBMITTED FILES

# USAGE:
# FOR SENTENCE-LEVEL TRACK:
# python validation.py -t 1 -i /your_submission/track1/inputs -o /your_submission/track1/outputs
# FOR DOCUMENT-LEVEL TRACK:
# python validation.py -t 2 -i /your_submission/track2/inputs -o /your_submission/track2/outputs

# EXAMPLE OF THE OUTPUT:
# IF EVERYTHING IS FINE, YOU WILL SEE THE FOLLOWING LINES:

# Test 1/3: checking filenames consistency...
# CHECK FILENAMES: DONE
# Test 2/3: checking consistency with source data...
# CHECK CONSISTENCY WITH SOURCE DATA: DONE
# Test 3/3: checking validity of data points...
# CHECK VALIDITY OF DATA POINTS: DONE
# All entries formated correctly. The files are ready for submission.

# OTHERWISE, YOU WILL SEE THE CORRESPONDING ERRORS POINTING AT THE FORMATTING PROBLEMS, FOR INSTANCE:
# ERROR: bad.2016.zhen.noterm.jsonl missing in submission
# ERROR: wrong key set in file output/bad/bad.enes.random.jsonl, 0 line: expected en, es, terms, got en, terms


import json
import os
import argparse


def file_basic_check(filepath, opener=False):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(l) for l in f.readlines()]
            if opener:
                return data
        return True
    except:
        return False


def file_naming_check(fname, track=1):
    fname_list = fname.split('.')
    if track == 1:
        # {system}.{src}{tgt}.{terminology mode}.jsonl
        assert len(fname_list) == 4 and fname_list[-1] == 'jsonl' and fname_list[-2] in ['noterm', 'random',
                                                                                         'proper'], f"ERROR: wrong naming of file {fname}, has to be sysname.lang_pair.mode.jsonl"
    elif track == 2:
        # {system}.{year}.{src}{tgt}.{terminology mode}.jsonl
        assert len(fname_list) == 5 and fname_list[-1] == 'jsonl' and fname_list[-2] in ['noterm', 'random',
                                                                                         'proper'], f"ERROR: wrong naming of file {fname}, has to be sysname.lang_pair.mode.jsonl"
    return None


def file_year_pair_check(fnames_list, sysname, track=1):
    submitted_pairs_candidates = set([f.split('.')[-3] for f in fnames_list])

    submitted_pairs = []
    if track == 1:
        for lang_pair in ['enes', 'enru', 'ende']:
            if lang_pair in submitted_pairs_candidates:
                submitted_pairs.append(lang_pair)
    elif track == 2:
        year2pair_candidates = {pair[0]: pair[1] for pair in zip(range(2015, 2025), ['enzh', 'zhen'] * 5)}
    expected_fnames = []
    if track == 1:
        for lang_pair in submitted_pairs:
            expected_fnames.extend([f'{sysname}.{lang_pair}.noterm.jsonl', f'{sysname}.{lang_pair}.random.jsonl',
                                    f'{sysname}.{lang_pair}.proper.jsonl'])
    elif track == 2:
        for year, lang_pair in year2pair_candidates.items():
            expected_fnames.extend(
                [f'{sysname}.{year}.{lang_pair}.noterm.jsonl', f'{sysname}.{year}.{lang_pair}.random.jsonl',
                 f'{sysname}.{year}.{lang_pair}.proper.jsonl'])

    for fname in expected_fnames:
        assert fname in fnames_list, f"ERROR: {fname} missing in submission"
    return None


def file_check(output_list, track=1):
    """
    Checks filenames for consistency in a list of output files:

    - Ensures that the provided list of file paths contains only valid `.jsonl` files,
    - verifies the naming consistency of the files: system name (same for all submissions), year (track 2 only), language pair, terminology mode.
    - ensures that the files submitted for a given language pair contain all three terminology modes (and all years for track 2).

    :param output_list: list[str], List of file paths to be checked.
    :param track: int, Track identifier (1 for sentence-level MT, 2 for document-level MT).
    :return: Confirmation message indicating filenames check completion.
    """

    print("Test 1/3: checking filenames consistency...")

    # retrieving only .jsonl submissions
    output_list = [f for f in output_list if f.endswith('.jsonl')]

    # checking the naming consistency
    fnames_list = [f.split(os.path.sep)[-1] for f in output_list]

    for fname in fnames_list:
        file_naming_check(fname, track=track)
    submitted_sysname_candidates = set([f.split('.')[0] for f in fnames_list])
    assert len(
        submitted_sysname_candidates) == 1, f"ERROR: inconsistent system naming: {', '.join(list(submitted_sysname_candidates))}"
    sysname = list(submitted_sysname_candidates)[0]

    # checking if all files are present for a particular language pair
    file_year_pair_check(fnames_list, sysname, track=track)

    # checking whether every file has valid JSONL encoding
    for filepath in output_list:
        assert file_basic_check(filepath), f"ERROR: {filepath} does not have JSONL format"

    return "CHECK FILENAMES: DONE"


def find_file_by_prefix(filename, file_list, is_input=False):
    if is_input:
        filename = '.'.join(filename.split('.')[1:])
    for file in file_list:
        if file.endswith(filename):
            return file


def sample_check(output_list, input_list):
    """
    Verifies the consistency between input and output data files.

    Includes:
    - validating the number of lines,
    - checking consistency between source sentences and terminology dictionaries in both input and output files.

    :param output_list: list[str], List of paths to output files.
    :param input_list: list[str], List of paths to input files used for cross-checking the
        consistency of data with the corresponding output files.
    :return: Completion message indicating the consistency checks have been successfully performed.
    """
    print('Test 2/3: checking consistency with source data...')
    fnames_list = [f.split(os.path.sep)[-1] for f in output_list if f.endswith('.jsonl')]

    for fname in fnames_list:
        langpair = fname.split('.')[-3]
        src, tgt = langpair[:2], langpair[2:]

        output_fname = find_file_by_prefix(fname, output_list)
        input_fname = find_file_by_prefix(fname, input_list, is_input=True)

        output_data = file_basic_check(output_fname, opener=True)
        input_data = file_basic_check(input_fname, opener=True)

        assert len(input_data) == len(
            output_data), f"ERROR: inconsistent number of lines in {output_fname}: expected {len(input_data)}, got {len(output_data)}"

        for idx in range(len(output_data)):
            input_datapoint, output_datapoint = input_data[idx], output_data[idx]
            input_src, output_src = input_datapoint[src], output_datapoint[src]
            input_terms = input_datapoint['terms'] if 'terms' in input_datapoint.keys() else {}
            output_terms = output_datapoint['terms'] if 'terms' in output_datapoint.keys() else {}

            assert input_src == output_src and input_terms == output_terms, f"ERROR: inconsistent input data (src sentences or terms), file {output_fname}, line {idx}"

    return "CHECK CONSISTENCY WITH SOURCE DATA: DONE"


def check_dict_format(terms, track=1):
    track2dtype = {1: str, 2: list}

    for key in terms.keys():
        if type(terms[key]) != track2dtype[track]:
            return False, type(terms[key])

    return True, None


def datapoint_check(output_list, track=1):
    """
    Checks the validity of each data point in JSONL files on whether it contains correct keys and value data types.

    - checks the key namings,
    - checks the value data types (especially for terminology dictionaries).

    :param output_list: list[str], A list containing paths to output files that need validation.
    :param track: int, Track identifier (1 for sentence-level MT, 2 for document-level MT)
    :return: A confirmation message indicating successful validation of all data points.
    """

    print('Test 3/3: checking validity of data points...')
    fnames_list = [f.split(os.path.sep)[-1] for f in output_list if f.endswith('.jsonl')]

    for fname in fnames_list:
        langpair = fname.split('.')[-3]
        src, tgt = langpair[:2], langpair[2:]

        mode = fname.split('.')[-2]

        output_fname = find_file_by_prefix(fname, output_list)

        data = file_basic_check(output_fname, opener=True)

        for idx, datapoint in enumerate(data):
            key_list = [src, tgt, 'terms'] if mode in ['random', 'proper'] else [src, tgt]
            assert set(key_list) == set(
                datapoint.keys()), f"ERROR: wrong key set in file {output_fname}, {idx} line: expected {', '.join(key_list)}, got {', '.join(list(datapoint.keys()))}"
            assert type(datapoint[
                            src]) == str, f"ERROR: wrong dtype of input sentence in file {output_fname}, {idx} line: expected str, got {type(datapoint[src])}"
            assert type(datapoint[
                            tgt]) == str, f"ERROR: wrong dtype of output sentence in file {output_fname}, {idx} line: expected str, got {type(datapoint[tgt])}"
            if mode in ['random', 'proper']:
                assert type(datapoint[
                                'terms']) == dict, f"ERROR: wrong dtype of terminology in file {output_fname}, {idx} line: expected dict, got {type(datapoint['terms'])}"
                is_dict_format_okay, observed_dtype = check_dict_format(datapoint['terms'], track=track)
                assert is_dict_format_okay, f"ERROR: wrong dtype of terminology values in file {output_fname}, {idx} line: expected {'str' if track == 1 else 'list(str)'}, got {observed_dtype}"
    return "CHECK VALIDITY OF DATA POINTS: DONE"


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-t", "--track", type=int, help="competition track: 1 for sentence-level MT, 2 for document-level MT")
    argparser.add_argument("-i", "--inputs", type=str, help="file path for input folder given track (no internal folders!)")
    argparser.add_argument("-o", "--outputs", type=str, help="file path for input folder for given track (no internal folders, all files for a given system)")
    args = argparser.parse_args()

    if args.track == 1:
        pass
    elif args.track == 2:
        pass
    else:
        raise Exception("you must choose a track: 1 for sentence-level MT, 2 for document-level MT")

    input_list = [f'{args.inputs}{os.path.sep}{f}' for f in os.listdir(args.inputs)]
    output_list = [f'{args.outputs}{os.path.sep}{f}' for f in os.listdir(args.outputs)]

    # 1. checking the file assortiment and formats
    file_check_message = file_check(output_list, track=args.track)
    print(file_check_message)
    # 2. checking the consistency with the input files
    sample_check_message = sample_check(output_list, input_list)
    print(sample_check_message)
    # 3. checking if all the keys are in place
    datapoint_check_message = datapoint_check(output_list, track=args.track)
    print(datapoint_check_message)
    print ("All entries formated correctly. The files are ready for submission.")

if __name__ == "__main__":
    main()