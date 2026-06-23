"""
=============================================================================
Live Correlation Processing (on-the-fly reduction)
=============================================================================

This module computes correlation / coincidence statistics on the TimeTagger itself (FPGA + driver) during the acquisition 
and stores only the reduced result (histograms and scalar counters) as a `.pkl`.

It plugs into the acquisition orchestrator through the same `Recorder` interface, 
so on-the-fly processing is opted into simply by handing a `CorrelationRecorder` to `Acquisition`. 
The orchestrator itself never imports this module: acquisition and processing stay decoupled.

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
"""

from __future__ import annotations

import logging
from itertools import combinations, combinations_with_replacement, permutations
from pathlib import Path
from typing import Optional

import numpy as np

from .acquisition import Recorder, StopCondition, save_pickle
from .hardware import TimeTaggerDevice

__all__ = [
    "CorrelationRecorder",
    "CombinedRecorder",
    "coincidence_threshold_stop",
]


# =============================================================================
# Recorder
# =============================================================================

class CorrelationRecorder(Recorder):
    """Recorder that performs live correlation/coincidence reduction.

    Parameters
    ----------
    mode : {'g2', 'g2_heralded_virtual', None}
        Reduction mode. If ``None`` (default) it is read from
        ``params['general']['experiment_type']`` at record time, so the single
        source of truth is the :class:`~src.acquisition.AcquisitionConfig`.
    record_physical_histograms : bool
        Also store the genuine per-detector-pair correlation histograms
        (``correlations_physical``). Recommended: it lets
        :class:`~src.core.HBTMeasurement` compute the artefact-free physical
        g^(2) by the peak-area method.
    write_raw : bool
        Also stream the raw photon time tags to a ``.ttbin`` alongside the reduced
        ``.pkl`` (see :class:`CombinedRecorder`). Off by default.
    max_file_size_mb : float
        Max size per ``.ttbin`` subfile when ``write_raw`` (0 = single file).
    """

    name = "correlation" 
    
    VIRTUAL_MODES = ("g2_heralded_virtual",)

    def __init__(self, mode: Optional[str] = None,
                 record_physical_histograms: bool = True,
                 write_raw: bool = False,
                 max_file_size_mb: float = 100.0):
        self.mode = mode
        self.record_physical_histograms = record_physical_histograms
        self.write_raw = write_raw
        self.max_file_size_mb = max_file_size_mb

    @staticmethod
    def build_modes(channels, mode_on_channel) -> "dict[str, list[int]]":
        """Group physical channels into harmonics by stripping the T/R suffix.

        e.g. channels=[1,2,3,4], modes=['H3T','H3R','H4T','H4R']
             -> {'H3': [1, 2], 'H4': [3, 4]}
        """
        modes: dict[str, list[int]] = {}
        for channel, mode_string in zip(channels, mode_on_channel):
            mode_name = mode_string[:-1]
            modes.setdefault(mode_name, []).append(channel)
        return modes

    def _resolve_mode(self, params: dict) -> str:
        return self.mode or params["general"]["experiment_type"]

    # -- main recording -------------------------------------------------------

    def record(self, device: TimeTaggerDevice, duration_ps: int, savepath: Path,
               *, params: dict, logger: logging.Logger) -> dict:
        import TimeTagger as TT

        tagger = device._require()
        tt = params["general"]["timetagging"]
        channels = list(tt["channels"])
        binwidth = int(tt["binwidth_ps"])
        num_bins = int(tt["num_bins"])
        coincidence_window = int(tt["coincidence_window"])
        mode_on_channel = list(tt["mode_on_channel"])
        mode = self._resolve_mode(params)
        do_virtual = mode in self.VIRTUAL_MODES

        logger.info("Correlation recording: mode=%s, virtual=%s, phys_hist=%s", mode, do_virtual, self.record_physical_histograms)

        results: dict = {
            "recorder": self.name,
            "mode": mode,
            "duration_ps": int(duration_ps),
            "counts_physical": {},
            "countrates_physical": {},
            "coincidences_twofold_physical": {},
        }
        if self.record_physical_histograms:
            results["correlations_physical"] = {}
        if do_virtual:
            results.update({
                "counts_virtual": {},
                "countrates_virtual": {},
                "coincidences_twofold_virtual": {},
                "correlations_virtual": {},
                "heralded_threefold": {},
            })

        # --- Virtual channel construction (kept alive for the whole run) -----
        _vc_store = []
        modes = mode_vcs = mode_names = herald_permutations = heralded_vcs = None
        if do_virtual:
            modes = self.build_modes(channels, mode_on_channel)
            logger.info("Harmonic modes: %s", modes)
            mode_vcs = {}
            for mode_name, phys in modes.items():
                combiner = TT.Combiner(tagger, phys)
                _vc_store.append(combiner)
                mode_vcs[mode_name] = combiner.getChannel()
            mode_names = list(mode_vcs.keys())

            herald_permutations = (list(permutations(mode_names, 3))
                                   if len(mode_names) >= 3 else [])
            heralded_vcs = {}
            for h_mode, s1_mode, _s2 in herald_permutations:
                key = f"h{h_mode}_s1{s1_mode}_s2{_s2}"
                coincidence_vc = TT.Coincidence(
                    tagger, [mode_vcs[h_mode], mode_vcs[s1_mode]],
                    coincidenceWindow=coincidence_window,
                    timestamp=TT.CoincidenceTimestamp_ListedFirst)
                _vc_store.append(coincidence_vc)
                heralded_vcs[key] = coincidence_vc

        file_writer = None
        ttbin_path = None
        with TT.SynchronizedMeasurements(tagger) as sync:
            sm = sync.getTagger()

            # Optionally stream raw tags to .ttbin in the SAME measurement group, so
            # the run yields both the reduced .pkl and the full raw .ttbin at once.
            if self.write_raw:
                ttbin_path = str(savepath.with_suffix(".ttbin"))
                file_writer = TT.FileWriter(sm, ttbin_path, channels)
                if self.max_file_size_mb > 0:
                    file_writer.setMaxFileSize(int(self.max_file_size_mb * 1024 * 1024))

            phys_counts = TT.Countrate(sm, channels)
            phys_pairs = list(combinations(channels, 2))
            phys_coinc_gen = TT.Coincidences(sm, coincidenceGroups=phys_pairs,
                                             coincidenceWindow=coincidence_window)
            phys_coinc_counts = TT.Countrate(sm, channels=phys_coinc_gen.getChannels())

            phys_corr = {}
            if self.record_physical_histograms:
                for c1, c2 in phys_pairs:
                    phys_corr[(c1, c2)] = TT.Correlation(sm, c1, c2, binwidth, num_bins)

            if do_virtual:
                virt_counts = TT.Countrate(sm, list(mode_vcs.values()))
                virt_pairs = list(combinations(list(mode_vcs.values()), 2))  # harmonic pairs
                virt_coinc_gen = TT.Coincidences(sm, coincidenceGroups=virt_pairs,
                                                 coincidenceWindow=coincidence_window)
                virt_coinc_counts = TT.Countrate(sm, channels=virt_coinc_gen.getChannels())

                virt_corr = {}
                for m1, m2 in combinations_with_replacement(mode_names, 2):
                    virt_corr[(m1, m2)] = TT.Correlation(
                        sm, mode_vcs[m1], mode_vcs[m2], binwidth, num_bins)

                h_num, h_den, h_rate = {}, {}, {}
                for h_mode, s1_mode, s2_mode in herald_permutations:
                    key = f"h{h_mode}_s1{s1_mode}_s2{s2_mode}"
                    hvc = heralded_vcs[key]
                    h_num[key] = TT.Correlation(sm, hvc.getChannel(),
                                                mode_vcs[s2_mode], binwidth, num_bins)
                    h_den[key] = TT.Correlation(sm, mode_vcs[h_mode],
                                                mode_vcs[s2_mode], binwidth, num_bins)
                    h_rate[key] = TT.Countrate(sm, [mode_vcs[h_mode], hvc.getChannel()])

            logger.info("Measuring for %.3f min ...", duration_ps * 1e-12 / 60)
            sync.startFor(int(duration_ps))
            sync.waitUntilFinished()

        for ch, count, rate in zip(channels, phys_counts.getCountsTotal(),
                                   phys_counts.getData()):
            results["counts_physical"][str(ch)] = int(count)
            results["countrates_physical"][str(ch)] = float(rate)

        for pair, count in zip(phys_pairs, phys_coinc_counts.getCountsTotal()):
            results["coincidences_twofold_physical"][f"({pair[0]},{pair[1]})"] = int(count)

        if self.record_physical_histograms:
            for (c1, c2), corr in phys_corr.items():
                results["correlations_physical"][f"({c1},{c2})"] = {
                    "time_bins": corr.getIndex().tolist(),
                    "counts": corr.getData().tolist(),
                }

        if do_virtual:
            for name, count, rate in zip(mode_names, virt_counts.getCountsTotal(),
                                         virt_counts.getData()):
                results["counts_virtual"][name] = int(count)
                results["countrates_virtual"][name] = float(rate)

            ch_to_mode = {ch: name for name, ch in mode_vcs.items()}
            for pair, count in zip(virt_pairs, virt_coinc_counts.getCountsTotal()):
                m1, m2 = ch_to_mode[pair[0]], ch_to_mode[pair[1]]
                results["coincidences_twofold_virtual"][f"({m1},{m2})"] = int(count)

            for (m1, m2), corr in virt_corr.items():
                results["correlations_virtual"][f"({m1},{m2})"] = {
                    "time_bins": corr.getIndex().tolist(),
                    "counts": corr.getData().tolist(),
                }

            for h_mode, s1_mode, s2_mode in herald_permutations:
                key = f"h{h_mode}_s1{s1_mode}_s2{s2_mode}"
                h_r, hs1_r = h_rate[key].getData()
                h_c, hs1_c = h_rate[key].getCountsTotal()
                results["heralded_threefold"][key] = {
                    "numerator": {
                        "time_bins": h_num[key].getIndex().tolist(),
                        "counts": h_num[key].getData().tolist(),
                    },
                    "denominator": {
                        "time_bins": h_den[key].getIndex().tolist(),
                        "counts": h_den[key].getData().tolist(),
                    },
                    "rates": {"herald": float(h_r), "heralded_s1": float(hs1_r)},
                    "counts": {"herald": int(h_c), "heralded_s1": int(hs1_c)},
                }

        if file_writer is not None:
            results["ttbin_file"] = ttbin_path
            results["total_events"] = int(file_writer.getTotalEvents())
            results["total_size"] = int(file_writer.getTotalSize())
            logger.info("Raw tags -> %s (%d events, %.2f MB)", ttbin_path,
                        results["total_events"], results["total_size"] / (1024 * 1024))
            del file_writer

        del _vc_store

        save_pickle({"Parameters": params, "data": results}, str(savepath) + ".pkl")
        logger.info("Saved correlation data -> %s.pkl", savepath)
        return results

    # -- chunk accumulation ---------------------------------------------------

    def merge(self, accumulated: Optional[dict], new: dict) -> dict:
        """Accumulate two reduced-result dicts (port of merge_experiment_data).

        Counts and coincidences are summed, histograms summed bin-by-bin,
        count rates duration-weighted. Works for both ``g2`` and
        ``g2_heralded_virtual`` payloads.
        """
        if accumulated is None:
            return _copy_result(new)

        d_acc = accumulated.get("duration_ps", 0) or 0
        d_new = new.get("duration_ps", 0) or 0
        total = (d_acc + d_new) or 1
        w_acc, w_new = d_acc / total, d_new / total

        merged = dict(accumulated)
        merged["duration_ps"] = d_acc + d_new

        for key in ("counts_physical", "counts_virtual",
                    "coincidences_twofold_physical", "coincidences_twofold_virtual"):
            if key in new:
                merged[key] = _sum_scalar_maps(accumulated.get(key, {}), new[key])

        for key in ("countrates_physical", "countrates_virtual"):
            if key in new:
                merged[key] = _weighted_maps(accumulated.get(key, {}), new[key],
                                             w_acc, w_new)

        for key in ("correlations_physical", "correlations_virtual"):
            if key in new:
                merged[key] = _sum_histogram_maps(accumulated.get(key, {}), new[key])

        if "heralded_threefold" in new:
            merged["heralded_threefold"] = _merge_heralded(
                accumulated.get("heralded_threefold", {}), new["heralded_threefold"],
                w_acc, w_new)

        # Raw .ttbin bookkeeping when running the combined recorder.
        if "ttbin_file" in new or "ttbin_files" in accumulated:
            files = list(accumulated.get("ttbin_files", []))
            if "ttbin_file" in accumulated and accumulated["ttbin_file"] not in files:
                files.append(accumulated["ttbin_file"])
            if "ttbin_file" in new:
                files.append(new["ttbin_file"])
            merged["ttbin_files"] = files
        for key in ("total_events", "total_size"):
            if key in new:
                merged[key] = accumulated.get(key, 0) + new[key]

        return merged


class CombinedRecorder(CorrelationRecorder):
    """Record BOTH the raw time tags (``.ttbin``) and the live-reduced correlation
    statistics (``.pkl``) from a single acquisition.

    It is exactly :class:`CorrelationRecorder` with ``write_raw=True``: the raw
    :class:`FileWriter` is attached to the same ``SynchronizedMeasurements`` group
    as the correlations, so one run produces, per chunk, a ``.ttbin`` (full raw
    stream) and a ``.pkl`` (histograms + scalars), and the chunked orchestrator
    additionally writes the usual ``MERGED/..._MERGED.pkl``.
    """

    name = "combined"

    def __init__(self, mode: Optional[str] = None,
                 record_physical_histograms: bool = True,
                 max_file_size_mb: float = 100.0):
        super().__init__(mode=mode,
                         record_physical_histograms=record_physical_histograms,
                         write_raw=True, max_file_size_mb=max_file_size_mb)


# =============================================================================
# Stop conditions
# =============================================================================

def coincidence_threshold_stop(min_counts: int,
                               kind: str = "physical") -> StopCondition:
    """Build a stop condition: stop once every two-fold coincidence pair of
    ``kind`` ('physical' or 'virtual') has reached ``min_counts``.

    Returns a callable ``(merged_data) -> (should_stop, reason)`` suitable for
    :class:`src.acquisition.Acquisition`'s ``stop_condition``.
    """
    store = f"coincidences_twofold_{kind}"

    def _stop(merged: dict) -> "tuple[bool, str]":
        coincidences = merged.get(store, {})
        if not coincidences:
            return False, f"no {store} yet"
        below = {k: v for k, v in coincidences.items() if v < min_counts}
        if not below:
            return True, f"all {kind} pairs >= {min_counts}"
        worst_k = min(below, key=below.get)
        return False, (f"{len(below)}/{len(coincidences)} {kind} pairs below "
                       f"{min_counts} (worst {worst_k}={below[worst_k]:,})")

    return _stop


# =============================================================================
# Merge primitives (module-level, easy to unit-test without hardware)
# =============================================================================

def _copy_result(result: dict) -> dict:
    out = {}
    for k, v in result.items():
        if isinstance(v, dict):
            out[k] = _copy_result(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _sum_scalar_maps(a: dict, b: dict) -> dict:
    keys = set(a) | set(b)
    return {k: a.get(k, 0) + b.get(k, 0) for k in keys}


def _weighted_maps(a: dict, b: dict, w_a: float, w_b: float) -> dict:
    keys = set(a) | set(b)
    return {k: a.get(k, 0.0) * w_a + b.get(k, 0.0) * w_b for k in keys}


def _sum_histogram_maps(a: dict, b: dict) -> dict:
    out = {}
    keys = set(a) | set(b)
    for k in keys:
        if k in a and k in b:
            out[k] = {
                "time_bins": a[k]["time_bins"],
                "counts": (np.asarray(a[k]["counts"]) + np.asarray(b[k]["counts"])).tolist(),
            }
        else:
            out[k] = a.get(k, b.get(k))
    return out


def _merge_heralded(a: dict, b: dict, w_a: float, w_b: float) -> dict:
    out = {}
    keys = set(a) | set(b)
    for k in keys:
        if k not in a:
            out[k] = b[k]
            continue
        if k not in b:
            out[k] = a[k]
            continue
        h1, h2 = a[k], b[k]
        out[k] = {
            "numerator": {
                "time_bins": h1["numerator"]["time_bins"],
                "counts": (np.asarray(h1["numerator"]["counts"])
                           + np.asarray(h2["numerator"]["counts"])).tolist(),
            },
            "denominator": {
                "time_bins": h1["denominator"]["time_bins"],
                "counts": (np.asarray(h1["denominator"]["counts"])
                           + np.asarray(h2["denominator"]["counts"])).tolist(),
            },
            "rates": {
                "herald": h1["rates"]["herald"] * w_a + h2["rates"]["herald"] * w_b,
                "heralded_s1": (h1["rates"]["heralded_s1"] * w_a
                                + h2["rates"]["heralded_s1"] * w_b),
            },
            "counts": {
                "herald": h1["counts"]["herald"] + h2["counts"]["herald"],
                "heralded_s1": h1["counts"]["heralded_s1"] + h2["counts"]["heralded_s1"],
            },
        }
    return out
