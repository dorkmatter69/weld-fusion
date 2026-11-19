"""One-weld audio feature and exploratory MCAP export in a Modal CPU Sandbox."""

import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import soundfile as sf

from weldfusion.dataset import WeldSample, load_sensor_table
from weldfusion.mcap_io import summarize_mcap, write_weld_mcap


SAMPLE_ID = "04-01-23-0024-00"
N_FFT = 1366  # 85.375 ms at 16 kHz; paper used 16,384 / 192 kHz = 85.333 ms.
HOP = 683     # 50% overlap, 42.6875 ms.


def download(url, path, max_bytes):
    try:
        with urllib.request.urlopen(url, timeout=60) as source, open(path, "wb") as target:
            size = 0
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise RuntimeError("pilot input exceeds size limit")
                target.write(block)
    except Exception:
        raise RuntimeError("pilot input download failed") from None
    return size


def make_audio_features(flac_path, output_path):
    data, rate = sf.read(str(flac_path), dtype="float32", always_2d=True)
    if rate != 16000 or data.shape != (608000, 1):
        raise RuntimeError("unexpected pilot FLAC sample count, rate, or channels")
    mono = data[:, 0]
    start = np.arange(0, len(mono) - N_FFT + 1, HOP, dtype=np.int32)
    frames = np.lib.stride_tricks.sliding_window_view(mono, N_FFT)[start]
    # This is a documented 16 kHz analysis, not the paper's 192 kHz input.
    magnitude = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1)).astype(np.float32)
    log_magnitude = np.log1p(magnitude)
    center_s = (start.astype(np.float64) + (N_FFT - 1) / 2) / rate
    if not np.isfinite(log_magnitude).all():
        raise RuntimeError("nonfinite audio features")
    np.savez_compressed(
        output_path, log_magnitude=log_magnitude,
        frame_start_sample=start, frame_center_seconds=center_s,
        frequency_hz=np.fft.rfftfreq(N_FFT, d=1 / rate),
        sample_rate_hz=np.int32(rate), sample_count=np.int32(len(mono)),
        n_fft=np.int32(N_FFT), hop_length=np.int32(HOP),
        transform=np.array("log1p(abs(rfft(hann_window * mono)))"),
        clock_domain=np.array("flac_start_elapsed_s"),
    )
    with np.load(output_path, allow_pickle=False) as saved:
        if (saved["log_magnitude"].shape != (len(start), N_FFT // 2 + 1)
                or not np.isfinite(saved["log_magnitude"]).all()):
            raise RuntimeError("audio feature archive failed validation")
    return {
        "sample_rate_hz": rate, "sample_count": len(mono),
        "audio_duration_s": len(mono) / rate,
        "stft_frames": len(start), "stft_bins": N_FFT // 2 + 1,
        "window_duration_ms": 1000 * N_FFT / rate,
        "hop_duration_ms": 1000 * HOP / rate,
        "first_center_s": float(center_s[0]),
        "last_center_s": float(center_s[-1]),
    }


def upload(url, path, key):
    payload = Path(path).read_bytes()
    request = urllib.request.Request(
        url, data=payload, method="PUT",
        headers={"Content-Type": "application/octet-stream",
                 "x-goog-if-generation-match": "0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status not in (200, 201):
                raise RuntimeError("unexpected upload status")
    except Exception:
        raise RuntimeError("pilot output upload failed; check create-only IAM and object existence") from None
    return {
        "object": key, "bytes": len(payload),
        "md5_base64": base64.b64encode(hashlib.md5(payload).digest()).decode("ascii"),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main():
    if os.environ["PILOT_SAMPLE_ID"] != SAMPLE_ID:
        raise RuntimeError("only the measured pilot weld is supported")
    urls = {ext: os.environ.pop("PILOT_GET_" + ext.upper())
            for ext in ("avi", "flac", "csv")}
    put_audio = os.environ.pop("PILOT_PUT_AUDIO")
    put_mcap = os.environ.pop("PILOT_PUT_MCAP")
    audio_key = os.environ["PILOT_AUDIO_OBJECT"]
    mcap_key = os.environ["PILOT_MCAP_OBJECT"]
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / SAMPLE_ID
        folder.mkdir()
        sizes = {}
        for ext, limit in (("avi", 100 * 1024 * 1024),
                           ("flac", 5 * 1024 * 1024), ("csv", 1024 * 1024)):
            sizes[ext] = download(urls[ext], folder / (SAMPLE_ID + "." + ext), limit)
        audio_file = folder / (SAMPLE_ID + ".flac")
        csv_file = folder / (SAMPLE_ID + ".csv")
        avi_file = folder / (SAMPLE_ID + ".avi")
        sensor = load_sensor_table(csv_file)
        if sensor.timestamps_s is None or len(sensor) != 335:
            raise RuntimeError("pilot sensor timestamps or row count changed")

        audio_path = Path(directory) / "pilot-audio.npz"
        audio_info = make_audio_features(audio_file, audio_path)
        sample = WeldSample(
            stem=SAMPLE_ID, label="unverified", class_id=None,
            date=None, run=None, video=avi_file, sensors=csv_file,
            audio=audio_file,
        )
        mcap_path = Path(directory) / "pilot-exploratory.mcap"
        alignment = write_weld_mcap(
            sample, mcap_path, embed_every=0, allow_unregistered_clocks=True
        )
        summary = summarize_mcap(mcap_path)
        required = {"/weld/meta", "/weld/alignment", "/sensors/weld",
                    "/video/brightness", "/audio/rms"}
        if not required.issubset(summary["topics"]):
            raise RuntimeError("pilot MCAP missing required streams")
        if (summary["meta"]["sensors"]["clock_alignment_verified"] is not False
                or summary["topics"]["/sensors/weld"]["count"] != 335
                or summary["topics"]["/video/brightness"]["count"] != 1119):
            raise RuntimeError("pilot MCAP failed verification")

        artifacts = [
            upload(put_audio, audio_path, audio_key),
            upload(put_mcap, mcap_path, mcap_key),
        ]
        print(json.dumps({
            "sample_id": SAMPLE_ID, "input_bytes": sizes,
            "audio_features": audio_info, "mcap_topics": summary["topics"],
            "event_candidates": {
                "sensor": alignment["arc_on_sensor_s"],
                "video": alignment["arc_on_video_s"],
                "audio": alignment["arc_on_audio_s"],
            },
            "clock_alignment_verified": False,
            "labels": {"category": None, "split": None,
                       "status": "manifest_not_audited"},
            "artifacts": artifacts,
        }), flush=True)


if __name__ == "__main__":
    main()
