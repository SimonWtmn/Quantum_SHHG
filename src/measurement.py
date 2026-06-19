"""
=============================================================================
Live Correlation Processing (on-the-fly reduction)
=============================================================================

The *processing* counterpart of the strict-acquisition layer. Where
:class:`src.acquisition.RawTimeTagRecorder` streams raw time tags to disk,
this module computes correlation / coincidence statistics **on the TimeTagger
itself** (FPGA + driver) during the acquisition and stores only the reduced
result - histograms and scalar counters - as a ``.pkl``.

It plugs into the acquisition orchestrator through the same
:class:`src.acquisition.Recorder` interface, so on-the-fly processing is opted
into simply by handing a :class:`CorrelationRecorder` to
:class:`src.acquisition.Acquisition`. The orchestrator itself never imports
this module: acquisition and processing stay decoupled.

This is an object-oriented port of the legacy
``experiment_utils.synchronized_correlation_measurement`` /
``merge_experiment_data`` / ``check_coincidence_threshold`` logic.

Supported reduction modes
--------------------------
* ``"g2"`` - per-detector (physical) singles, all-pair two-fold coincidences
  and (optionally) all-pair correlation histograms.
* ``"g2_heralded_virtual"`` - everything in ``"g2"`` **plus** merged-harmonic
  *virtual* channels (T+R combined per harmonic), virtual singles /
  coincidences / correlations, and full three-fold *heralded* correlations for
  every herald permutation.

The output dictionary uses exactly the key schema consumed by
:class:`src.hbt_core.HBTMeasurement` (``counts_physical``,
``correlations_physical``, ``correlations_virtual``, ``heralded_threefold``,
...), so acquired data is analysable downstream with no conversion.

Author: Simon WITTMANN
Institution: Laboratoire d'Optique Appliquee (LOA), Ecole Polytechnique
"""

# Allow type hints that name a class inside its own definition.
from __future__ import annotations

import logging  # for log messages during the measurement
# itertools provides ready-made combinatorics generators:
#   combinations([1,2,3], 2)                -> (1,2), (1,3), (2,3)        [unordered pairs]
#   combinations_with_replacement(..., 2)   -> also includes (1,1), (2,2) [self-pairs too]
#   permutations([A,B,C], 3)                -> all ORDERED triples (A,B,C),(A,C,B),...
from itertools import combinations, combinations_with_replacement, permutations
from pathlib import Path        # filesystem paths
from typing import Optional     # type hint: "either a value or None"

import numpy as np  # used to add histogram arrays together when merging

# Reuse the Recorder base class, the StopCondition type alias and the pickle
# helper from the acquisition module (so we save in the same format/place).
from .acquisition import Recorder, StopCondition, save_pickle
from .hardware import TimeTaggerDevice  # type hint for the device we receive

# Public names exported by this module.
__all__ = [
    "CorrelationRecorder",
    "coincidence_threshold_stop",
]


# =============================================================================
# Recorder
# =============================================================================

# CorrelationRecorder IS-A Recorder: it can be plugged into Acquisition exactly
# like RawTimeTagRecorder, but it computes correlations instead of saving raw tags.
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
        :class:`~src.hbt_core.HBTMeasurement` compute the artefact-free physical
        g^(2) by the peak-area method.
    """

    name = "correlation"  # identifies this recorder in logs

    # Modes that additionally build "virtual" (merged-harmonic) channels.
    VIRTUAL_MODES = ("g2_heralded_virtual",)

    def __init__(self, mode: Optional[str] = None,
                 record_physical_histograms: bool = True):
        self.mode = mode                                       # None = take mode from config
        self.record_physical_histograms = record_physical_histograms  # store per-pair histograms?

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def build_modes(channels, mode_on_channel) -> "dict[str, list[int]]":
        """Group physical channels into harmonics by stripping the T/R suffix.

        e.g. channels=[1,2,3,4], modes=['H3T','H3R','H4T','H4R']
             -> {'H3': [1, 2], 'H4': [3, 4]}
        """
        modes: dict[str, list[int]] = {}  # will map "H3" -> [1, 2], etc.
        # Walk each channel with its label (e.g. channel 1 labelled "H3T").
        for channel, mode_string in zip(channels, mode_on_channel):
            mode_name = mode_string[:-1]   # drop the last char (T/R) -> harmonic name "H3"
            # setdefault: create an empty list the first time we see this harmonic,
            # then append this channel to it.
            modes.setdefault(mode_name, []).append(channel)
        return modes

    def _resolve_mode(self, params: dict) -> str:
        # Use the explicit mode if given, else fall back to the config's experiment_type.
        return self.mode or params["general"]["experiment_type"]

    # -- main recording -------------------------------------------------------

    def record(self, device: TimeTaggerDevice, duration_ps: int, savepath: Path,
               *, params: dict, logger: logging.Logger) -> dict:
        import TimeTagger as TT  # lazy import of the vendor library

        tagger = device._require()                 # connected hardware handle
        tt = params["general"]["timetagging"]      # the tagger settings block
        channels = list(tt["channels"])            # detector channels, e.g. [1..6]
        binwidth = int(tt["binwidth_ps"])          # histogram bin width (ps)
        num_bins = int(tt["num_bins"])             # number of histogram bins
        coincidence_window = int(tt["coincidence_window"])  # coincidence window (ps)
        mode_on_channel = list(tt["mode_on_channel"])       # per-channel labels
        mode = self._resolve_mode(params)          # which reduction mode to run
        do_virtual = mode in self.VIRTUAL_MODES    # whether to build virtual channels

        logger.info("Correlation recording: mode=%s, virtual=%s, phys_hist=%s",
                    mode, do_virtual, self.record_physical_histograms)

        # Pre-create the result dict with the exact keys the analysis code reads.
        # Physical = real detectors; values keyed by channel/pair as strings.
        results: dict = {
            "recorder": self.name,
            "mode": mode,
            "duration_ps": int(duration_ps),
            "counts_physical": {},                  # total singles per detector
            "countrates_physical": {},              # singles per second per detector
            "coincidences_twofold_physical": {},    # 2-fold coincidence totals per pair
        }
        if self.record_physical_histograms:
            results["correlations_physical"] = {}   # full histograms per detector pair
        if do_virtual:
            # Virtual = merged harmonics (T+R combined). Add the extra result slots.
            results.update({
                "counts_virtual": {},
                "countrates_virtual": {},
                "coincidences_twofold_virtual": {},
                "correlations_virtual": {},
                "heralded_threefold": {},           # 3-fold heralded correlations
            })

        # --- Virtual channel construction (kept alive for the whole run) -----
        # `_vc_store` holds references to virtual-channel objects so Python's
        # garbage collector does not delete them mid-measurement.
        _vc_store = []
        # Pre-declare names so they exist even when do_virtual is False.
        modes = mode_vcs = mode_names = herald_permutations = heralded_vcs = None
        if do_virtual:
            modes = self.build_modes(channels, mode_on_channel)  # {"H3":[1,2], ...}
            logger.info("Harmonic modes: %s", modes)
            mode_vcs = {}  # maps harmonic name -> the virtual channel number it produces
            for mode_name, phys in modes.items():
                # Combiner merges several physical channels into ONE virtual channel
                # (so H3 fires whenever detector 1 OR 2 fires).
                combiner = TT.Combiner(tagger, phys)
                _vc_store.append(combiner)              # keep it alive
                mode_vcs[mode_name] = combiner.getChannel()  # the new virtual channel id
            mode_names = list(mode_vcs.keys())  # e.g. ["H3", "H4", "H5"]

            # Heralding needs 3 distinct modes: herald (h), signal-1 (s1), signal-2 (s2).
            # Build every ORDERED triple of modes (only if we have at least 3).
            herald_permutations = (list(permutations(mode_names, 3))
                                   if len(mode_names) >= 3 else [])
            heralded_vcs = {}  # maps a herald-permutation key -> its coincidence channel
            for h_mode, s1_mode, _s2 in herald_permutations:
                key = f"h{h_mode}_s1{s1_mode}_s2{_s2}"  # unique label for this permutation
                # A Coincidence virtual channel fires only when h AND s1 fire within the window.
                # timestamp=ListedFirst means it is timestamped at the first listed channel (h).
                coincidence_vc = TT.Coincidence(
                    tagger, [mode_vcs[h_mode], mode_vcs[s1_mode]],
                    coincidenceWindow=coincidence_window,
                    timestamp=TT.CoincidenceTimestamp_ListedFirst)
                _vc_store.append(coincidence_vc)        # keep it alive
                heralded_vcs[key] = coincidence_vc

        # --- Measurement objects under one synchronized group ----------------
        # All measurements created inside this `with` start and stop together.
        with TT.SynchronizedMeasurements(tagger) as sync:
            sm = sync.getTagger()  # a "proxy" tagger that ties measurements to this group

            phys_counts = TT.Countrate(sm, channels)             # singles on real detectors
            phys_pairs = list(combinations(channels, 2))          # all detector pairs (15 for 6)
            # Coincidences builds a virtual channel for each pair that fires on a coincidence.
            phys_coinc_gen = TT.Coincidences(sm, coincidenceGroups=phys_pairs,
                                             coincidenceWindow=coincidence_window)
            # Count how often each of those pair-coincidence channels fired.
            phys_coinc_counts = TT.Countrate(sm, channels=phys_coinc_gen.getChannels())

            phys_corr = {}  # maps (c1,c2) -> its Correlation measurement object
            if self.record_physical_histograms:
                for c1, c2 in phys_pairs:
                    # Correlation builds the full time-difference histogram for the pair.
                    phys_corr[(c1, c2)] = TT.Correlation(sm, c1, c2, binwidth, num_bins)

            if do_virtual:
                virt_counts = TT.Countrate(sm, list(mode_vcs.values()))  # singles per harmonic
                virt_pairs = list(combinations(list(mode_vcs.values()), 2))  # harmonic pairs
                virt_coinc_gen = TT.Coincidences(sm, coincidenceGroups=virt_pairs,
                                                 coincidenceWindow=coincidence_window)
                virt_coinc_counts = TT.Countrate(sm, channels=virt_coinc_gen.getChannels())

                virt_corr = {}  # maps (mode1,mode2) -> Correlation object
                # combinations_with_replacement also includes self-pairs (H3,H3) for autocorrelation.
                for m1, m2 in combinations_with_replacement(mode_names, 2):
                    virt_corr[(m1, m2)] = TT.Correlation(
                        sm, mode_vcs[m1], mode_vcs[m2], binwidth, num_bins)

                # Three-fold heralded correlations: for each (h, s1, s2) permutation we need
                #   numerator   = correlation of (h AND s1 coincidence) with s2
                #   denominator = correlation of h with s2          (for normalisation)
                #   rate        = counts of herald and of the herald+s1 coincidence
                h_num, h_den, h_rate = {}, {}, {}
                for h_mode, s1_mode, s2_mode in herald_permutations:
                    key = f"h{h_mode}_s1{s1_mode}_s2{s2_mode}"
                    hvc = heralded_vcs[key]  # the h&s1 coincidence channel built earlier
                    h_num[key] = TT.Correlation(sm, hvc.getChannel(),
                                                mode_vcs[s2_mode], binwidth, num_bins)
                    h_den[key] = TT.Correlation(sm, mode_vcs[h_mode],
                                                mode_vcs[s2_mode], binwidth, num_bins)
                    h_rate[key] = TT.Countrate(sm, [mode_vcs[h_mode], hvc.getChannel()])

            logger.info("Measuring for %.3f min ...", duration_ps * 1e-12 / 60)
            sync.startFor(int(duration_ps))  # run all the above for the requested time
            sync.waitUntilFinished()          # block until finished

        # --- Collect physical ------------------------------------------------
        # getCountsTotal() = totals; getData() = rates (counts/s). zip pairs them per channel.
        for ch, count, rate in zip(channels, phys_counts.getCountsTotal(),
                                   phys_counts.getData()):
            results["counts_physical"][str(ch)] = int(count)
            results["countrates_physical"][str(ch)] = float(rate)

        # Store the integrated coincidence count for each detector pair, keyed "(c1,c2)".
        for pair, count in zip(phys_pairs, phys_coinc_counts.getCountsTotal()):
            results["coincidences_twofold_physical"][f"({pair[0]},{pair[1]})"] = int(count)

        if self.record_physical_histograms:
            for (c1, c2), corr in phys_corr.items():
                # getIndex() = the time-axis (ps); getData() = counts per bin. Save as lists.
                results["correlations_physical"][f"({c1},{c2})"] = {
                    "time_bins": corr.getIndex().tolist(),
                    "counts": corr.getData().tolist(),
                }

        # --- Collect virtual -------------------------------------------------
        if do_virtual:
            # Virtual singles per harmonic, keyed by harmonic name ("H3", ...).
            for name, count, rate in zip(mode_names, virt_counts.getCountsTotal(),
                                         virt_counts.getData()):
                results["counts_virtual"][name] = int(count)
                results["countrates_virtual"][name] = float(rate)

            # Reverse map: virtual channel number -> harmonic name, to label pairs.
            ch_to_mode = {ch: name for name, ch in mode_vcs.items()}
            for pair, count in zip(virt_pairs, virt_coinc_counts.getCountsTotal()):
                m1, m2 = ch_to_mode[pair[0]], ch_to_mode[pair[1]]
                results["coincidences_twofold_virtual"][f"({m1},{m2})"] = int(count)

            # Virtual correlation histograms, keyed "(H3,H4)" etc.
            for (m1, m2), corr in virt_corr.items():
                results["correlations_virtual"][f"({m1},{m2})"] = {
                    "time_bins": corr.getIndex().tolist(),
                    "counts": corr.getData().tolist(),
                }

            # Heralded three-fold results, one block per herald permutation.
            for h_mode, s1_mode, s2_mode in herald_permutations:
                key = f"h{h_mode}_s1{s1_mode}_s2{s2_mode}"
                h_r, hs1_r = h_rate[key].getData()           # rates: herald, herald&s1
                h_c, hs1_c = h_rate[key].getCountsTotal()    # totals: herald, herald&s1
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

        del _vc_store  # release virtual channels (Combiners/Coincidences) now we're done

        # Save the reduced data in the standard {"Parameters":..., "data":...} layout.
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
            # First chunk: just keep a fresh copy as the running total.
            return _copy_result(new)

        d_acc = accumulated.get("duration_ps", 0) or 0  # time accumulated so far
        d_new = new.get("duration_ps", 0) or 0           # time of the new chunk
        total = (d_acc + d_new) or 1                     # combined time (never 0)
        w_acc, w_new = d_acc / total, d_new / total      # weights for averaging rates

        merged = dict(accumulated)                       # start from the old total
        merged["duration_ps"] = d_acc + d_new            # update the total duration

        # Totals (counts, coincidences): add element-by-element.
        for key in ("counts_physical", "counts_virtual",
                    "coincidences_twofold_physical", "coincidences_twofold_virtual"):
            if key in new:
                merged[key] = _sum_scalar_maps(accumulated.get(key, {}), new[key])

        # Rates: duration-weighted average (a long chunk counts more than a short one).
        for key in ("countrates_physical", "countrates_virtual"):
            if key in new:
                merged[key] = _weighted_maps(accumulated.get(key, {}), new[key],
                                             w_acc, w_new)

        # Histograms: add the two count arrays bin-by-bin (keeping the shared time axis).
        for key in ("correlations_physical", "correlations_virtual"):
            if key in new:
                merged[key] = _sum_histogram_maps(accumulated.get(key, {}), new[key])

        # Heralded three-fold blocks have their own nested structure: merge separately.
        if "heralded_threefold" in new:
            merged["heralded_threefold"] = _merge_heralded(
                accumulated.get("heralded_threefold", {}), new["heralded_threefold"],
                w_acc, w_new)

        return merged


# =============================================================================
# Stop conditions
# =============================================================================

# This is a "factory": you call it once with a threshold, and it RETURNS a
# function. That returned function is what Acquisition calls after each chunk.
def coincidence_threshold_stop(min_counts: int,
                               kind: str = "physical") -> StopCondition:
    """Build a stop condition: stop once every two-fold coincidence pair of
    ``kind`` ('physical' or 'virtual') has reached ``min_counts``.

    Returns a callable ``(merged_data) -> (should_stop, reason)`` suitable for
    :class:`src.acquisition.Acquisition`'s ``stop_condition``.
    """
    store = f"coincidences_twofold_{kind}"  # which dict in the merged data to inspect

    # The inner function "remembers" min_counts and store (this is a closure).
    def _stop(merged: dict) -> "tuple[bool, str]":
        coincidences = merged.get(store, {})  # {pair: total_count}
        if not coincidences:                  # nothing measured yet -> keep going
            return False, f"no {store} yet"
        # Find which pairs are still below the required number of counts.
        below = {k: v for k, v in coincidences.items() if v < min_counts}
        if not below:                         # none below -> all good -> stop
            return True, f"all {kind} pairs >= {min_counts}"
        worst_k = min(below, key=below.get)   # the pair with the fewest counts
        return False, (f"{len(below)}/{len(coincidences)} {kind} pairs below "
                       f"{min_counts} (worst {worst_k}={below[worst_k]:,})")

    return _stop  # hand back the configured stop function


# =============================================================================
# Merge primitives (module-level, easy to unit-test without hardware)
# =============================================================================

def _copy_result(result: dict) -> dict:
    # Recursively copy a result dict so later merges don't mutate the original.
    out = {}
    for k, v in result.items():
        if isinstance(v, dict):
            out[k] = _copy_result(v)   # recurse into nested dicts
        elif isinstance(v, list):
            out[k] = list(v)           # copy lists (so they're independent)
        else:
            out[k] = v                 # numbers/strings are immutable: copy by value
    return out


def _sum_scalar_maps(a: dict, b: dict) -> dict:
    # Add two {key: number} dicts together; missing keys count as 0.
    keys = set(a) | set(b)             # the union of all keys in either dict
    return {k: a.get(k, 0) + b.get(k, 0) for k in keys}


def _weighted_maps(a: dict, b: dict, w_a: float, w_b: float) -> dict:
    # Weighted average of two {key: number} dicts (used for count rates).
    keys = set(a) | set(b)
    return {k: a.get(k, 0.0) * w_a + b.get(k, 0.0) * w_b for k in keys}


def _sum_histogram_maps(a: dict, b: dict) -> dict:
    # Add two {key: {"time_bins":..., "counts":...}} histogram dicts.
    out = {}
    keys = set(a) | set(b)
    for k in keys:
        if k in a and k in b:          # present in both -> add counts bin-by-bin
            out[k] = {
                "time_bins": a[k]["time_bins"],  # time axis is identical, keep one copy
                "counts": (np.asarray(a[k]["counts"]) + np.asarray(b[k]["counts"])).tolist(),
            }
        else:                          # present in only one -> take whichever exists
            out[k] = a.get(k, b.get(k))
    return out


def _merge_heralded(a: dict, b: dict, w_a: float, w_b: float) -> dict:
    # Merge the nested "heralded_threefold" blocks permutation by permutation.
    out = {}
    keys = set(a) | set(b)
    for k in keys:
        if k not in a:                 # only in the new chunk -> keep it as-is
            out[k] = b[k]
            continue
        if k not in b:                 # only in the old total -> keep it as-is
            out[k] = a[k]
            continue
        h1, h2 = a[k], b[k]            # both present -> combine field by field
        out[k] = {
            # numerator/denominator are histograms: add counts bin-by-bin.
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
            # rates are averages -> duration-weighted; counts are totals -> summed.
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
