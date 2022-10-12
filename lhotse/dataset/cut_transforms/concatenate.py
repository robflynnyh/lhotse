from typing import List, Optional, Sequence

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.utils import Seconds

def isfalse(val):
    return val == False

class CutConcatenate:
    """
    A transform on batch of cuts (``CutSet``) that concatenates the cuts to minimize the total amount of padding;
    e.g. instead of creating a batch with 40 examples, we will merge some of the examples together
    adding some silence between them to avoid a large number of padding frames that waste the computation.
    """

    def __init__(self, gap: Seconds = 1.0, duration_factor: float = 1.0) -> None:
        """
        CutConcatenate's constructor.

        :param gap: The duration of silence in seconds that is inserted between the cuts;
            it's goal is to let the model "know" that there are separate utterances in a single example.
        :param duration_factor: Determines the maximum duration of the concatenated cuts;
            by default it's 1, setting the limit at the duration of the longest cut in the batch.
        """
        self.gap = gap
        self.duration_factor = duration_factor

    def __call__(self, cuts: CutSet) -> CutSet:
        cuts = cuts.sort_by_duration(ascending=False)
        return concat_cuts(
            cuts, gap=self.gap, max_duration=cuts[0].duration * self.duration_factor
        )


def concat_cuts(
    cuts: Sequence[Cut], gap: Seconds = 1.0, max_duration: Optional[Seconds] = None
) -> CutSet:
    """
    We're going to concatenate the cuts to minimize the amount of total padding frames used.
    This means that some samples in the batch will be merged together into one sample,
    separated by an interval of silence.
    This is actually solving a knapsack problem.
    In this initial implementation we're using a greedy approach:
    going from the back (i.e. the shortest cuts) we'll try to concat them to the longest cut
    that still has some "space" at the end.

    :param cuts: a list of cuts to pack.
    :param gap: the duration of silence inserted between concatenated cuts.
    :param max_duration: the maximum duration for the concatenated cuts
        (by default set to the duration of the first cut).
    :return a list of packed cuts.
    """
    if len(cuts) <= 1:
        # Nothing to do.
        return CutSet.from_cuts(cuts)
    cuts = sorted(cuts, key=lambda c: c.duration, reverse=True)
    max_duration = cuts[0].duration if max_duration is None else max_duration
    current_idx = 0
    while True:
        can_fit = False
        shortest = cuts[-1]
        for idx in range(current_idx, len(cuts) - 1):
            cut = cuts[current_idx]
            can_fit = cut.duration + gap + shortest.duration <= max_duration
            if can_fit:
                cuts[current_idx] = cut.pad(cut.duration + gap).append(shortest)
                cuts = cuts[:-1]
                break
            current_idx += 1
        if not can_fit:
            break
    return CutSet.from_cuts(cuts)

def individual_speaker_concat(cuts: Sequence[Cut], gap: Seconds = 0.1, speaker_gap: Seconds=1.0, speaker_list:List[str]=[], max_duration=None, concat_cuts=True):
    '''
    Conatenate cuts, speakers are joined into a single sample up to max duration, if there is a gap with another speaker present silence of speaker_gap is added
    cuts: a lhotse sequence of cuts
    gap: the duration of silence inserted between adjacent concatenated cuts
    speaker_gap: the silence duration inserted between concatenated cuts when there was a speaker change
    speaker_list: list of speakers for each cut sample in the cut sequence
    max_duration: the maximum duration of the concatenated cut (seconds). If None, no limit is applied.
    concat_cuts: whether to concatenate cuts or just join into individual cutsets and return a list of cutsets
    '''
    if len(cuts) <= 1:
        # Nothing to do.
        return [CutSet.from_cuts(cuts)]
    assert len(cuts) == len(speaker_list), "speaker list must be same length as cuts"
    max_duration = max_duration if max_duration is not None else float('inf')
    gap, speaker_gap = (gap, speaker_gap) if concat_cuts else (0.0, 0.0)

    cutdata = {}
    cutdata['prev_speaker'] = speaker_list[0]

    cutdata[cutdata['prev_speaker']] = {
        'cuts':[cuts[0]] if concat_cuts else [[cuts[0]]],
        'cur_duration':cuts[0].duration,
    }

    for idx in range(1, len(cuts)):
        cut = cuts[idx]
        speaker = speaker_list[idx]
        gapsize = gap if speaker == cutdata['prev_speaker'] else speaker_gap

        if speaker not in cutdata:
            cutdata[speaker] = {
                'cuts':[cut] if concat_cuts else [[cut]],
                'cur_duration':cut.duration,
            }
        elif (cutdata[speaker]['cur_duration'] + cut.duration + gapsize) <= max_duration:
            cutdata[speaker]['cuts'][-1] = cutdata[speaker]['cuts'][-1].pad(cutdata[speaker]['cuts'][-1].duration + gapsize).append(cut) \
                if concat_cuts else cutdata[speaker]['cuts'][-1] + [cut]
            cutdata[speaker]['cur_duration'] += cut.duration + gapsize
        else:
            cutdata[speaker]['cuts'].append(cut) if concat_cuts else cutdata[speaker]['cuts'].append([cut])
            cutdata[speaker]['cur_duration'] = cut.duration
        cutdata['prev_speaker'] = speaker

    #return cutdata
    
    all_cuts = []
    for speaker in cutdata:
        if speaker == 'prev_speaker':
            continue
        all_cuts += cutdata[speaker]['cuts']
    
    return [CutSet.from_cuts([cut] if concat_cuts else cut) for cut in all_cuts]
           

    

def plain_concat(cuts: Sequence[Cut], gap: Seconds = 0.1, max_duration=None, seperate_speakers=False, concat_cuts=True, speaker_list:List[str]=[]) -> CutSet:
    """
    A simple concatenation of cuts, maintaining original order.
    cuts: a lhotse sequence of cuts
    gap: the duration of silence inserted between concatenated cuts.
    max_duration: the maximum duration of the concatenated cut (seconds). If None, no limit is applied.
    seperate_speakers: whether to include seperate speakers in the same utterance, pass list of speakers to seperate
    concat_cuts: whether to concatenate cuts or just join into individual cutsets and return a list of cutsets
    """
    if len(cuts) <= 1:
        # Nothing to do.
        return [CutSet.from_cuts(cuts)]
    assert isfalse(seperate_speakers) or speaker_list.__class__.__name__ == 'list' and len(speaker_list) > 0, "seperate_speakers must be a list of speakers to seperate" 
 
    max_duration = max_duration if max_duration is not None else float('inf')
    gap = gap if concat_cuts else 0.0 # no gap if not concatenating

    cutlist = [cuts[0]] if concat_cuts else [[cuts[0]]]
    prev_speaker = None if isfalse(seperate_speakers) else speaker_list[0]
    cur_duration = cuts[0].duration 

    for i in range(1, len(cuts)):
        cut = cuts[i]
        cur_speaker = None if isfalse(seperate_speakers) else speaker_list[i]

        if cur_speaker == prev_speaker and (cur_duration + gap + cut.duration) <= max_duration:
            cutlist[-1] = cutlist[-1].pad(cutlist[-1].duration + gap).append(cut) if concat_cuts else cutlist[-1] + [cut]
            cur_duration += gap + cut.duration
        else:
            cutlist.append(cut if concat_cuts else [cut])
            cur_duration = cut.duration

        prev_speaker = cur_speaker

    return [CutSet.from_cuts([cut] if concat_cuts else cut) for cut in cutlist]


