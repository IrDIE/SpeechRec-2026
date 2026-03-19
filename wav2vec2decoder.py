import math
from typing import List, Tuple
import time
import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from clearml import Task, Logger
from utils import read_test_samples, default_samples
import os
import hashlib
import numpy as np
import pickle, heapq

# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


# decorator to wrap methods and evaluate tima of execution (letency)


def decorator_latency(return_time=False):
    def timeit(method):
        def timed(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            delta = end_time - start_time
            print(f"{method.__name__} took {delta:.4f} seconds")
            if return_time:
                return result, delta
            return result

        return timed

    return timeit


class Wav2Vec2Decoder:
    def __init__(
        self,
        model_name="facebook/wav2vec2-base-100h",
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=3,
        alpha=1.0,
        beta=1.0,
        temperature=1.0,
    ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = "".join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, " ").strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    @decorator_latency(return_time=True)
    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            torch.Size([760, 32])
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        max_probs = torch.argmax(logits, dim=-1)  # (T,)
        res = self._ids_to_text(max_probs.tolist())
        return res

    @decorator_latency(return_time=True)
    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        T, V = logits.size()
        log_probs = logits.cpu().numpy()
        beams = [(tuple(), 0.0)]
        K = 15

        for t in range(T):
            ac_scores = log_probs[t]
            top_k_indices = np.argpartition(-ac_scores, K)[:K]
            new_candidates = []
            for prefix, score in beams:

                for v in top_k_indices:
                    new_score = score + ac_scores[v]
                    new_prefix = prefix + (v,)
                    new_candidates.append((new_prefix, new_score))

            merged = {}
            for prefix, score in new_candidates:
                if prefix not in merged or score > merged[prefix]:
                    merged[prefix] = score
            best_items = heapq.nlargest(
                self.beam_width, merged.items(), key=lambda x: x[1]
            )
            beams = [(prefix, score) for prefix, score in best_items]

        if return_beams:
            return [(list(prefix), score) for prefix, score in beams]

        best_prefix, best_score = max(beams, key=lambda x: x[1])
        return self._ids_to_text(list(best_prefix))

    @decorator_latency(return_time=True)
    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        T, V = logits.size()
        log_probs = logits.cpu().numpy()
        beams = [([], "", 0.0)]
        K = 15
        lm_cache = {}

        for t in range(T):
            new_beams = []
            ac_scores = log_probs[t]

            for prefix_ids, prefix_text, score in beams:

                top_indices = np.argpartition(-ac_scores, K)[:K]
                for v in top_indices:
                    acoustic_score = score + ac_scores[v]
                    new_ids = prefix_ids + [v]

                    if v == self.blank_token_id:

                        new_beams.append((new_ids, prefix_text, acoustic_score))
                        continue

                    token_txt = self.vocab[v]
                    new_text = prefix_text + (" " if prefix_text else "") + token_txt

                    if new_text not in lm_cache:
                        lm_cache[new_text] = self.lm_model.score(
                            new_text, bos=False, eos=False
                        )
                    lm_score = lm_cache[new_text]

                    total = (
                        acoustic_score
                        + self.alpha * lm_score
                        + self.beta * len(new_ids)
                    )
                    new_beams.append((new_ids, new_text, total))

            merged = {}
            for ids, txt, scr in new_beams:
                key = tuple(ids)
                if key not in merged or scr > merged[key][2]:
                    merged[key] = (ids, txt, scr)

            beams = heapq.nlargest(self.beam_width, merged.values(), key=lambda x: x[2])

        best = max(beams, key=lambda x: x[2])
        return best[1]

    @decorator_latency(return_time=True)
    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        rescored_beams = []
        for token_ids, log_prob in beams:
            # Convert token IDs to text
            text = self._ids_to_text(token_ids)
            # Compute LM score
            lm_score = self.lm_model.score(text, bos=False, eos=False)
            # Combine scores with alpha and beta
            total_score = log_prob + self.alpha * lm_score + self.beta * len(token_ids)
            rescored_beams.append((text, total_score))

        # Select the best rescored beam
        best_beam = max(rescored_beams, key=lambda x: x[1])
        return best_beam[0]

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def _get_cache_path(self, audio_path: str) -> str:
        """
        Generate a unique cache path for the given audio file path.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            str: Path to the cache file.
        """
        cache_dir = "/mnt/d/ITMO/2026-SpeechRec/hw2/logits_cache/"
        os.makedirs(cache_dir, exist_ok=True)
        audio_hash = hashlib.md5(audio_path.encode()).hexdigest()
        return os.path.join(cache_dir, f"{audio_hash}.pkl")

    def _load_logits_from_cache(self, cache_path: str) -> torch.Tensor:
        """
        Load logits from the cache.

        Args:
            cache_path (str): Path to the cache file.

        Returns:
            torch.Tensor: Cached logits.
        """
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    def _save_logits_to_cache(self, cache_path: str, logits: torch.Tensor) -> None:
        """
        Save logits to the cache.

        Args:
            cache_path (str): Path to the cache file.
            logits (torch.Tensor): Logits to save.
        """
        with open(cache_path, "wb") as f:
            pickle.dump(logits, f)

    def decode(
        self,
        audio_input: torch.Tensor = None,
        method: str = "greedy",
        audio_path: str = None,
    ) -> str:
        """
        Run the full decoding pipeline on a raw audio tensor or audio file.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz. torch.Size([1, 243520])
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".
            audio_path (str): Path to the audio file.

        Returns:
            str: Decoded transcript (lowercase).
        """

        cache_path = None
        if audio_path is not None:
            cache_path = self._get_cache_path(audio_path)

        if cache_path and os.path.exists(cache_path):
            print(f"Loading logits from cache: {cache_path}")
            logits = self._load_logits_from_cache(cache_path)
        else:
            print(
                f"Cache not found. Forwarding model and saving logits to cache: {cache_path}"
            )
            if audio_input is None and audio_path is not None:
                audio_input, sr = torchaudio.load(audio_path)
                assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"
            inputs = self.processor(
                audio_input, return_tensors="pt", sampling_rate=16000
            )
            with torch.no_grad():
                logits = self.model(inputs.input_values.squeeze(0)).logits[0]
            if cache_path:
                self._save_logits_to_cache(cache_path, logits)

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = torch.log_softmax(logits / self.temperature, dim=-1)

        if method == "greedy":
            result, latency = self.greedy_decode(logits)
        elif method == "beam":
            result, latency = self.beam_search_decode(logits)
        elif method == "beam_lm":
            result, latency = self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams, latency1 = self.beam_search_decode(logits, return_beams=True)
            result, latency2 = self.lm_rescore(beams)
            latency = latency1 + latency2

        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )

        return result.lower(), latency


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------


def test_single_sample(
    decoder: Wav2Vec2Decoder,
    audio_path: str,
    reference: str,
    logger: Logger,
    methods_test=["greedy", "beam", "beam_lm", "beam_lm_rescore"],
    log_self_attr="beam_width",
) -> dict:
    """
    Test a single audio sample and return metrics for each method.

    Args:
        decoder (Wav2Vec2Decoder): Decoder instance.
        audio_path (str): Path to the audio file.
        reference (str): Reference transcript.
        logger (Logger): ClearML logger.
        methods_test (list): List of decoding methods to test.
        log_self_attr (str): Attribute to log (e.g., beam_width, temperature).

    Returns:
        dict: Metrics for each method (WER, CER, Latency).
    """
    import jiwer

    # audio_input, sr = torchaudio.load(audio_path)
    # assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    metrics = {}
    for method in methods_test:
        try:
            hyp, latency = decoder.decode(
                audio_input=None, method=method, audio_path=audio_path
            )
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        log_xx = getattr(decoder, log_self_attr, None)
        if log_self_attr == "temperature":
            log_xx = int(
                decoder.temperature * 10
            )  # e.g., 0.5 -> 5, 1.2 -> 12 for better plotting
        # logger.report_scalar(title=f"{method}_metrics", series="WER", value=wer, iteration=log_xx)
        # logger.report_scalar(title=f"{method}_metrics", series="CER", value=cer, iteration=log_xx)
        # logger.report_scalar(title=f"{method}_metrics", series="Latency", value=latency, iteration=log_xx)
        metrics[method] = {"WER": wer, "CER": cer, "Latency": latency}
    return metrics


def test_multiple_samples(
    decoder: Wav2Vec2Decoder,
    test_samples: List[Tuple[str, str]],
    logger: Logger,
    methods_test=["greedy", "beam", "beam_lm", "beam_lm_rescore"],
    log_self_attr="beam_width",
) -> None:
    """
    Test multiple audio samples and log mean metrics.

    Args:
        decoder (Wav2Vec2Decoder): Decoder instance.
        test_samples (list): List of (audio_path, reference) tuples.
        logger (Logger): ClearML logger.
        methods_test (list): List of decoding methods to test.
        log_self_attr (str): Attribute to log (e.g., beam_width, temperature).
    """
    mean_metrics = {
        method: {"WER": 0, "CER": 0, "Latency": 0} for method in methods_test
    }
    valid_counts = {method: 0 for method in methods_test}

    for audio_path, reference in test_samples:
        metrics = test_single_sample(
            decoder, audio_path, reference, logger, methods_test, log_self_attr
        )
        for method, values in metrics.items():
            mean_metrics[method]["WER"] += values["WER"]
            mean_metrics[method]["CER"] += values["CER"]
            mean_metrics[method]["Latency"] += values["Latency"]
            valid_counts[method] += 1

    for method in methods_test:
        if valid_counts[method] > 0:
            mean_metrics[method]["WER"] /= valid_counts[method]
            mean_metrics[method]["CER"] /= valid_counts[method]
            mean_metrics[method]["Latency"] /= valid_counts[method]
            log_xx = getattr(decoder, log_self_attr, None)
            series_wer = "Mean_WER"
            series_cer = "Mean_CER"
            series_latency = "Mean_Latency"
            if log_self_attr == "temperature":
                log_xx = int(decoder.temperature * 10)
            elif log_self_attr == "alpha":
                log_xx = int(decoder.alpha * 10)
                # for alpha in x axis we transform metric name to considet betha
                series_wer = f"Mean_WER_b_{decoder.beta}"
                series_cer = f"Mean_CER_b_{decoder.beta}"
                series_latency = f"Mean_Latency_b_{decoder.beta}"

            logger.report_scalar(
                title=f"{method}_metrics",
                series=series_wer,
                value=mean_metrics[method]["WER"],
                iteration=log_xx,
            )
            logger.report_scalar(
                title=f"{method}_metrics",
                series=series_cer,
                value=mean_metrics[method]["CER"],
                iteration=log_xx,
            )
            logger.report_scalar(
                title=f"{method}_metrics",
                series=series_latency,
                value=mean_metrics[method]["Latency"],
                iteration=log_xx,
            )


def test_bim_width_latency(test_samples=None):

    task = Task.init(
        project_name="SpeechRec-2026/hw2", task_name="Beam Width Latency Test"
    )
    logger = task.get_logger()
    if test_samples is None:
        test_samples = default_samples()

    temperature = 1
    beam_w = 1
    decoder = Wav2Vec2Decoder(
            lm_model_path=None, beam_width=beam_w, temperature=temperature
        )
    for beam_w in [1, 3, 10, 50]:
        decoder.beam_width = beam_w
        print(f"Testing beam width {beam_w}...")
        test_multiple_samples(
            decoder, test_samples, logger=logger, methods_test=["beam"]
        )
    task.close()


def test_T_impact_grredy(test_samples=None):
    task = Task.init(project_name="SpeechRec-2026/hw2", task_name="Temp*10 Test")
    logger = task.get_logger()
    if test_samples is None:
        test_samples = default_samples()

    beam_w = 3
    for temperature in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        decoder = Wav2Vec2Decoder(
            lm_model_path=None, beam_width=beam_w, temperature=temperature
        )
        print(f"Testing temperature {temperature}...")
        test_multiple_samples(
            decoder,
            test_samples,
            logger=logger,
            methods_test=["greedy"],
            log_self_attr="temperature",
        )
    task.close()


def test_shallow_lm_fusion(test_samples=None):
    task = Task.init(
        project_name="SpeechRec-2026/hw2", task_name="Shallow LM Fusion Test"
    )
    logger = task.get_logger()
    if test_samples is None:
        test_samples = default_samples()

    beam_w = 3
    temperature = 1.0
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:  #  1.5, 2.0
        for beta in [0.5, 1.0, 1.5]:  # 0.5,  2.0
            decoder = Wav2Vec2Decoder(
                lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
                beam_width=beam_w,
                temperature=temperature,
                alpha=alpha,
                beta=beta,
            )
            print(f"Testing alpha {alpha} and beta {beta}...")
            test_multiple_samples(
                decoder,
                test_samples,
                logger=logger,
                methods_test=["beam_lm"],
                log_self_attr="alpha",
            )
    task.close()

def test_shallow_lm_fusion_T(test_samples=None):
    task = Task.init(
        project_name="SpeechRec-2026/hw2", task_name="Best Shallow LM Fusion Test Temp"
    )
    logger = task.get_logger()
    if test_samples is None:
        test_samples = default_samples()

    beam_w = 3
    temperature = 1.0
    decoder = Wav2Vec2Decoder(
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=beam_w,
            temperature=temperature,
            alpha=5,
            beta=0.5,
        )
    for temperature in [0.5, 1.0, 1.5, 2.0]:  #  1.5, 2.0
        decoder.temperature = temperature
        print(f"Testing alpha {temperature} temperature...")
        test_multiple_samples(
            decoder,
            test_samples,
            logger=logger,
            methods_test=["greedy","beam",  "beam_lm"],
            log_self_attr="temperature",
        )
    task.close()


def test_rescore_lm(test_samples=None):
    task = Task.init(
        project_name="SpeechRec-2026/hw2",
        task_name="Rescore LM Test",
    )
    logger = task.get_logger()
    if test_samples is None:
        test_samples = default_samples()

    beam_w = 3
    temperature = 1.0
    alpha = 1.0
    beta = 1.0
    decoder = Wav2Vec2Decoder(
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=beam_w,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
    )
    for alpha in [0. , 0.1, 0.5, 1.0, 2.0, 5.0]:  #  
        for beta in [0. , 0.5, 1.0, 1.5]:  # 0.5,  2.0
            decoder.alpha = alpha
            decoder.beta = beta
            print(f"Testing alpha {alpha} and beta {beta}...")
            test_multiple_samples(
                decoder,
                test_samples,
                logger=logger,
                methods_test=["beam_lm_rescore"],
                log_self_attr="alpha",
            )
    task.close()


def test_4_gramm(test_samples):
    
    task = Task.init(project_name="SpeechRec-2026/hw2", task_name="4 gramm beam LM Test")
    logger = task.get_logger()
    path_4_gramm = "/mnt/d/ITMO/2026-SpeechRec/4-gram.bin" #"/mnt/d/ITMO/2026-SpeechRec/4-gram.arpa.gz"
    beam_w = 3
    alpha = 5
    beta = 0.5

    decoder = Wav2Vec2Decoder(
        lm_model_path=path_4_gramm, beam_width=beam_w, alpha=alpha, beta=beta
    )
    print(f"Testing alpha {alpha} and beta {beta}...")
    test_multiple_samples(
        decoder,
        test_samples,
        logger=logger,
        methods_test=["beam_lm"],
        log_self_attr="alpha",
    )


if __name__ == "__main__":
    samples_path = "data/librispeech_test_other/manifest.csv"
    earnings22_test = "data/earnings22_test/manifest.csv"
    samples_test = read_test_samples(samples_path)
    # earnings22_test_samples = read_test_samples(earnings22_test)
    # samples_test = default_samples()
    # test_T_impact_grredy(samples_test)
    # test_bim_width_latency(samples_test)

    # test_shallow_lm_fusion(samples_test)
    # test_rescore_lm(test_samples=samples_test)
    test_4_gramm(samples_test)
    # test_shallow_lm_fusion_T(test_samples=earnings22_test_samples)


    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    # test_samples = [
    #     ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
    #     ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
    #     ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
    #     ("examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
    #     ("examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
    #     ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
    #     ("examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
    #     ("examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    # ]

    # decoder = Wav2Vec2Decoder(lm_model_path=None)  # set lm_model_path for Tasks 4+

    # for audio_path, reference in test_samples:
    #     test(decoder, audio_path, reference)
