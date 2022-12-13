"""
About the Earnings 22 dataset:

    The Earnings 22 dataset ( also referred to as earnings22 ) is a 119-hour corpus
    of English-language earnings calls collected from global companies. The primary
    purpose is to serve as a benchmark for industrial and academic automatic speech
    recognition (ASR) models on real-world accented speech.

    This dataset has been submitted to Interspeech 2022. The paper describing our
    methods and results can be found on arXiv at https://arxiv.org/abs/2203.15591.

    @misc{https://doi.org/10.48550/arxiv.2203.15591,
    doi = {10.48550/ARXIV.2203.15591},
    url = {https://arxiv.org/abs/2203.15591},
    author = {Del Rio, Miguel and Ha, Peter and McNamara, Quinten and Miller, Corey and Chandra, Shipra},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Earnings-22: A Practical Benchmark for Accents in the Wild},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution Share Alike 4.0 International}
    }

"""

from tqdm import tqdm
import logging
import string
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import os

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
import pandas as pd
import re
from lhotse import CutSet   
from lhotse.cut import MonoCut

_DEFAULT_URL = "https://github.com/revdotcom/speech-datasets"

'''
def normalize(text: str) -> str:
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert all upper case to lower case
    text = text.lower()
    return text
''' # I want to keep apostrophes

possible_denominations = ['million', 'thousand', 'billion', 'trillion', 'hundred', 'hundred thousand', 'ten thousand', '']
#possible_money_tokens = ["$","£",""]
money_fns = {
    '$': 'dollars',
    '£': 'pounds',
    '€': 'euros',
    '¥': 'yen',
    '₹': 'rupees',
    '₩': 'won',
    ' GH ': ' ghanacedis ', # replace ghanacedis with ghana cedis afterwards lol (makes the regex easier)
    'GH¢': 'ghanacedis',
}

def monetary_tokens(currency_token:str, replacement_string:str, text:str):
    '''
    Example usage:
    text = monetary_tokens('$', 'dollars', 'I have $100 million')
    output = 'I have 100 million dollars'
    too dumb for fancy regex so this will do lmao
    '''
    # first replace the currency token with the replacement string
    text = text.replace(currency_token, replacement_string + ' ')
    stext = text.split()
    # now loop through the text and carry the replacement string forward
    ops = []
    replacement_string = replacement_string.strip() # remove any whitespace
    for i, word in enumerate(stext):
        if word == replacement_string:
            ops.append({'from':i})
            to = i
            if i+1 < len(stext) and stext[i+1].isdigit():
                to = i+1
            if i+2 < len(stext) and stext[i+2] in possible_denominations:
                to = i+2
            ops[-1]['to'] = to
    # now loop through the ops and replace the tokens
    for op in ops:
        tomove = stext[op['from']]
        stext[op['from']] = ''
        stext[op['to']] = stext[op['to']] + ' ' + tomove
    return ' '.join(stext).strip()
    

def edge_cases(text: str):
    text = text.replace("&", " and ")
    text = text.replace("%", " percent ")
    text = text.replace(" ,", ",")
    text = text.replace('*', 'star')
    text = text.replace(',', '')
    return text

def convert_money_tokens(text: str):
    for token, replacement in money_fns.items():
        text = monetary_tokens(token, replacement, text)
    return text
    
earnings_junk_tokens = ["<noise>", "<crosstalk>", "<affirmative>", "<inaudible>", "inaudible", "<laugh>", "<silence>"]

def remove_junk_tokens(text: str):
    for token in earnings_junk_tokens:
        text = text.replace(token, "")
    return text

def normalize(text: str) -> str:
    '''
    accounts for:
    - percentage signs
    - Numbers
    '''
    import number_conversion # https://github.com/robflynnyh/number_conversion
    # lower
    # seperate GH\d from digits i.e GH100 -> GH 100 (exists as a monetary token) but GH is also sometimes used as a prefix i.e GHG
    text = re.sub(r"GH(\d)", r"GH \1", text)
    text = remove_junk_tokens(text)
    text = text.lower()
    # account for edge case

    text = edge_cases(text)

    # monetary tokens
    text = convert_money_tokens(text)
    # we want to keep unknown tokens so add some identifier that won't get removed by the regex
    text = text.replace("<unk>", "unkunkunk")
    # convert numbers to words
    text = number_conversion.convert_doc(text)
    # remove everything except letters and apostrophes
    text = re.sub(r"[^a-z']", " ", text)
    # add back the unk token
    text = text.replace("unkunkunk", "<unk>")
    # remove extra spaces
    text = re.sub(r"\s+", " ", text)
    # remove leading and trailing spaces
    text = text.strip()
    return text


def read_metadata(path: Pathlike) -> Dict[str, List[str]]:
    with open(path) as f:
        f.readline()  # skip header
        out = dict()
        for line in f:
            line = line.split(",")
            out[line[0]] = line[1:-1]
        return out


def download_earnings22(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    url: Optional[str] = _DEFAULT_URL,
) -> Path:
    """Download and untar the dataset.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
        The extracted files are saved to target_dir/earnings22/
        Please note that the github repository contains other additional datasets and
        using this call, you will be downloading all of them and then throwing them out.
    :param force_download: Bool, if True, download the tar file no matter
        whether it exists or not.
    :param url: str, the url to download the dataset.
    :return: the path to downloaded and extracted directory with data.
    """
    logging.error(
        "Downloading Earnings22 from github repository is not implemented. "
        + f"Please visit {_DEFAULT_URL} and download the files manually. Please "
        + "follow the instructions closely as you need to use git-lfs to download "
        + "some of the audio files."
    )


def parse_nlp_file(filename: Pathlike):
    with open(filename) as f:
        transcript = list()
        f.readline()  # skip header
        for line in f:
            line = line.split("|")
            transcript.append(line[0])
        return transcript

def get_empty_field_dict():
    return {
        'token': [],
        'speaker': [],
        'start': [],
        'end': [],
    }

def parse_nlp_fields(filename: Pathlike):
    fields = {
        'token': [],
        'speaker': [],
        'start': [],
        'end': [],
    }
    with open(filename) as f:
        f.readline()  # skip header
        for line in f:
            line = line.split("|")
            fields['token'].append(line[0])
            fields['speaker'].append(line[1])
            if line[2].strip() == '' or line[3].strip() == '':
                return False # return early if there are empty fields as we won't use this file
            fields['start'].append(float(line[2]))
            fields['end'].append(float(line[3]))
        return fields

def split_on_timings(fields: Dict[str, List[Any]], max_duration: float):
    """
    Split the fields on the timings. We will also split on the speaker change
    """
    utterances = [get_empty_field_dict()]
    cur_duration = 0
    prev_speaker = None
    for i in range(len(fields['token'])):
        duration = fields['end'][i] - fields['start'][i]
        prev_speaker = fields['speaker'][i] if prev_speaker is None else prev_speaker
        if cur_duration + duration > max_duration or prev_speaker != fields['speaker'][i]:
            utterances.append(get_empty_field_dict())
            cur_duration = 0
        utterances[-1]['token'].append(fields['token'][i])
        utterances[-1]['speaker'].append(fields['speaker'][i])
        utterances[-1]['start'].append(fields['start'][i])
        utterances[-1]['end'].append(fields['end'][i])
        cur_duration += duration
        prev_speaker = fields['speaker'][i]
    return utterances

def segments_to_supervisions(
    segments: List[Dict[str, List[Any]]],
    language: str,
    id: str,
    normalize_text: bool = True,
    ) -> List[SupervisionSegment]:
    """
    Convert List of utterances to List of Supervision Segments
    """
    out = []
    for i, utterance in enumerate(segments):
        text = " ".join(utterance['token'])
        if normalize_text:
            text = normalize(text)
        out.append(
            SupervisionSegment(
                id=f"{id}-{i}",
                recording_id=id,
                start=utterance['start'][0],
                duration=utterance['end'][-1] - utterance['start'][0],
                channel=0,
                language=language,
                speaker=utterance['speaker'][0],
                text=text,
                custom={ # otherwise lhotse discards timings after recordings are cut
                    'segment_start': utterance['start'][0],
                    'segment_end': utterance['end'][-1],
                },
            )
        )
    return out



def prepare_earnings22(
    split: str,
    df_path: Pathlike,
    data_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    normalize_text: bool = True,
) -> Union[RecordingSet, SupervisionSet]:
    '''
    df_path: path to the csv linking the wav files to their transcripts
    data: path to the directory containing the wav files 
    normalize_text: whether to normalize the text or not (see fn)
    '''
    df = pd.read_csv(df_path)
    startf, endf = 'start_ts', 'end_ts'
    textf = 'sentence'
    filef = 'file'
    sourcef = 'source_id'
    output_dir = Path(output_dir) if output_dir is not None else Path(data_dir)
    data_dir = Path(data_dir)
  
    

    
    cuts = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text, start, end = row[textf], row[startf], row[endf]

        f = data_dir / row[filef]
        r = Recording.from_file(f)
        r.id = f.parent.name + '-' + f.stem

        
        id, recording_id = r.id, r.id
        if normalize_text:
            text = normalize(text)

        supervision = SupervisionSegment(
                id=id,
                recording_id=recording_id,
                start=0,
                duration=end - start,
                channel=0,
                language='en',
                speaker='speaker',
                text=text,
                custom={ # otherwise lhotse discards timings after recordings are cut
                    'segment_start': start,
                    'segment_end': end,
                    'parent_id': row[sourcef]
                },
            )
        
        cut = MonoCut(
            id=r.id,
            start=0,
            duration=r.duration,
            channel=0,
            recording=r,
            supervisions=[supervision],
        )
        cuts.append(cut)

        
    cuts = CutSet.from_cuts(cuts)    

    print(f'Found {len(cuts)} cuts')

    if output_dir is not None:
        if os.path.exists(output_dir / split) is False:
            os.mkdir(output_dir / split)
        cuts.to_file(output_dir / f'earnings22_cuts_{split}.json.gz')
        # load the cuts
        cuts = CutSet.from_file(output_dir / f'earnings22_cuts_{split}.json.gz')
        print(f'Saved cuts')
        print(f'Number of cuts: {len(cuts)}')

        print(f'Writing utterances to {output_dir / split}')
        cuts = cuts.save_audios(storage_path=output_dir / split)
        # resave the cuts
        cuts.to_file(output_dir / f'earnings22_cuts_{split}.json.gz')
    print('OKAY BYE')


def __prepare_earnings22(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    normalize_text: bool = False,
    max_utterance_duration: float = 12.0
) -> Union[RecordingSet, SupervisionSet]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply
    read and return them.

    :param corpus_dir: Pathlike, the path of the data dir. The structure is
        expected to mimic the structure in the github repository, notably
        the mp3 files will be searched for in [corpus_dir]/media and transcriptions
        in the directory [corpus_dir]/transcripts/nlp_references
    :param output_dir: Pathlike, the path where to write the manifests.
    :param normalize_text: Bool, if True, normalize the text.
    :return: (recordings, supervisions) pair

    .. caution::
        The `normalize_text` option removes all punctuation and converts all upper case
        to lower case. This includes removing possibly important punctuations such as
        dashes and apostrophes.
    """

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    media_dir = corpus_dir / "media"
    audio_files = list(media_dir.glob("*.mp3"))
    assert len(audio_files) == 125

    audio_files.sort()
    recording_set = RecordingSet.from_recordings(
        Recording.from_file(p) for p in audio_files
    )

    nlp_dir = corpus_dir / "transcripts" / "nlp_references"
    nlp_files = list(nlp_dir.glob("*.nlp"))
    assert len(nlp_files) == 125

    metadata = read_metadata(corpus_dir / "metadata.csv")

    nlp_files.sort()
    supervision_segments = list()
    skipped = 0
    for nlp_file in nlp_files:
        id = nlp_file.stem
        #text = " ".join(parse_nlp_file(nlp_file))
        fields = parse_nlp_fields(nlp_file)
        # check all the fields are the same length
        if fields is False:
            logging.warning(f"Skipping {id} as it has inconsistent fields (we need timings)")
            skipped += 1
            continue

        if normalize_text:
            text = normalize(text)

        recording_supervisions = segments_to_supervisions(
            segments = split_on_timings(fields=fields, max_duration=max_utterance_duration),
            language=f"English-{metadata[id][4]}",
            id=id,
            normalize_text=normalize_text,
        )
        supervision_segments.extend(recording_supervisions)
    print(len(nlp_files))
    logging.warning(f"Skipped {skipped / len(nlp_files) * 100:.2f}% of the files due to missing timings")

    supervision_set = SupervisionSet.from_segments(supervision_segments)

    validate_recordings_and_supervisions(recording_set, supervision_set)
    if output_dir is not None:
        supervision_set.to_file(output_dir / "earnings22_supervisions_all.jsonl.gz")
        recording_set.to_file(output_dir / "earnings22_recordings_all.jsonl.gz")

    return recording_set, supervision_set
