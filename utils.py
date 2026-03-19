import csv
from typing import List, Tuple

def read_test_samples(csv_path: str) -> List[Tuple[str, str]]:
    """
    Read a CSV file and return test samples in the required format.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        List[Tuple[str, str]]: List of (audio_path, reference_text) tuples.
    """
    test_samples = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            audio_path = row['path']
            reference_text = row['text']
            test_samples.append((audio_path, reference_text))
    return test_samples

def default_samples():
    return  [
        ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
    ]

def find_samples_with_lm_changes(
    decoder, 
    test_samples: List[Tuple[str, str]], 
    num_samples: int = 10
) -> List[dict]:
    """
    Find samples where at least one LM method changes the hypothesis relative to plain beam search.

    Args:
        decoder: Wav2Vec2Decoder instance.
        test_samples (List[Tuple[str, str]]): List of (audio_path, reference_text) tuples.
        num_samples (int): Number of samples to return.

    Returns:
        List[dict]: List of dictionaries containing audio_path, reference, and hypotheses for each method.
    """
    results = []

    for audio_path, reference in test_samples:
        plain_beam, _ = decoder.decode(audio_path=audio_path, method="beam")
        beam_with_lm, _ = decoder.decode(audio_path=audio_path, method="beam_lm")
        rescored_lm, _ = decoder.decode(audio_path=audio_path, method="beam_lm_rescore")
        if plain_beam != beam_with_lm or plain_beam != rescored_lm:
            results.append({
                "audio_path": audio_path,
                "reference": reference,
                "plain_beam": plain_beam,
                "beam_with_lm": beam_with_lm,
                "rescored_lm": rescored_lm
            })

        # Stop if we have enough samples
        if len(results) >= num_samples:
            break

    return results