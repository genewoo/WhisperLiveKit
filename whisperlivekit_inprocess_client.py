#!/usr/bin/env python3
"""Sample in-process transcription client for WhisperLiveKit.

This script mirrors the behaviour of ``whisperlivekit_ws_client.py`` but keeps
all audio handling inside the current process instead of streaming through a
WebSocket server.

It captures audio from either an FFmpeg microphone input or a file, feeds raw
PCM frames to :class:`~whisperlivekit.AudioProcessor`, and renders incremental
transcriptions with Rich.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import AsyncGenerator, Iterable

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from whisperlivekit import AudioProcessor, TranscriptionEngine

console = Console()

LOG_PATH = Path(os.environ.get("SAMPLE_WHISPER_LOG", "sample_whisper.log"))

logger = logging.getLogger("sample_whisper")
if not logger.handlers:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

###############################################################################
# Configuration
###############################################################################
SAMPLE_RATE = int(os.environ.get("WLK_SAMPLE_RATE", "16000"))
CHANNELS = int(os.environ.get("WLK_CHANNELS", "1"))
SAMPLE_WIDTH = 2  # 16-bit PCM
PCM_CHUNK_SECONDS = float(os.environ.get("WLK_CHUNK_SECONDS", "0.5"))
PCM_CHUNK_BYTES = max(int(SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * PCM_CHUNK_SECONDS), SAMPLE_WIDTH)

DEFAULT_BACKEND = os.environ.get("WLK_BACKEND", "simulstreaming")
DEFAULT_MODEL_SIZE = os.environ.get("WLK_MODEL_SIZE", "base")
DEFAULT_LANGUAGE = os.environ.get("WLK_LANGUAGE", "en")
DEFAULT_ENABLE_DIARIZATION = os.environ.get("WLK_DIARIZATION", "false").lower() in {"1", "true", "yes"}
DEFAULT_ENABLE_VAC = os.environ.get("WLK_VAC", "true").lower() not in {"0", "false", "no"}
DEFAULT_MIN_CHUNK_SIZE = float(os.environ.get("WLK_MIN_CHUNK_SECONDS", PCM_CHUNK_SECONDS))

DEFAULT_MIC_DEVICE = os.environ.get("WLK_MIC_DEVICE", ":0")  # macOS avfoundation default
FILE_CHUNK_SECONDS = float(os.environ.get("WLK_FILE_CHUNK_SECONDS", "0.0"))


###############################################################################
# Helper functions
###############################################################################
def ffmpeg_cmd_for_mic(avfoundation_device: str | None = None) -> list[str]:
    """Return FFmpeg command line to capture microphone as raw PCM."""
    device = avfoundation_device or DEFAULT_MIC_DEVICE
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        os.environ.get("WLK_MIC_FORMAT", "avfoundation"),
        "-i",
        device,
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]


def ffmpeg_cmd_for_file(path: str) -> list[str]:
    """Return FFmpeg command line to transcode a file to raw PCM."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if FILE_CHUNK_SECONDS > 0:
        cmd.extend(["-re", "-async", "1"])
    cmd.extend([
        "-i",
        path,
        "-ac",
        str(CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ])
    return cmd


async def stream_pcm_to_processor(proc: asyncio.subprocess.Process, processor: AudioProcessor) -> None:
    """Read PCM bytes from ``proc.stdout`` and forward them to the processor."""
    assert proc.stdout is not None
    buffer = bytearray()
    try:
        while True:
            chunk = await proc.stdout.read(PCM_CHUNK_BYTES)
            if not chunk:
                break
            buffer.extend(chunk)
            while len(buffer) >= PCM_CHUNK_BYTES:
                frame = bytes(buffer[:PCM_CHUNK_BYTES])
                del buffer[:PCM_CHUNK_BYTES]
                await processor.process_audio(frame)
        if buffer:
            await processor.process_audio(bytes(buffer))
    finally:
        await processor.process_audio(b"")


async def _drain_stderr(proc: asyncio.subprocess.Process, label: str) -> None:
    """Log stderr output from FFmpeg to avoid blocking pipes."""
    if proc.stderr is None:
        return
    try:
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            if text:
                logger.info("%s stderr: %s", label, text)
    except Exception:  # pragma: no cover - logging best effort
        pass


async def recv_results(results: AsyncGenerator) -> None:
    """Consume transcription updates and render them with Rich."""
    status_display = "waiting"
    buffer_text = ""
    info_message = ""
    segments_state: list[dict] = []
    display_segments_state: list[dict] = []
    last_render_signature: list[tuple[int | None, str | None, str | None, str]] = []

    def _format_time_component(value: str | None) -> str:
        if value in (None, ""):
            return "?"
        return str(value)

    def _format_speaker(value: int | None) -> str:
        if value in (None, -2):
            return "--"
        try:
            return f"{int(value):02d}"
        except (TypeError, ValueError):
            return str(value)

    def _segment_signature(segment: dict) -> tuple[int | None, str | None, str | None, str]:
        return (
            segment.get("speaker"),
            segment.get("start"),
            segment.get("end"),
            segment.get("text", ""),
        )

    def _build_segments(lines: list[dict]) -> list[dict]:
        segments: list[dict] = []
        for line in lines:
            text_value = (line.get("text") or "").strip()
            if not text_value:
                continue

            speaker = line.get("speaker")
            start = line.get("start")
            end = line.get("end")

            if segments and segments[-1].get("speaker") == speaker:
                segment = segments[-1]
                if segment.get("start") in (None, "") and start:
                    segment["start"] = start
                if end:
                    segment["end"] = end
                segment["text"] = text_value
            else:
                segments.append(
                    {
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                        "text": text_value,
                    }
                )
        return segments

    def _build_renderable(
        segments: list[dict],
        status: str,
        buffer_msg: str,
        info_msg: str,
    ) -> Group:
        header = Text(f"Status: {status}", style="bold magenta")

        table = Table(
            show_header=True,
            header_style="bold white",
            expand=True,
            box=box.SIMPLE_HEAVY,
            pad_edge=False,
        )
        table.add_column("Speaker", justify="center", style="cyan", no_wrap=True)
        table.add_column("Start", style="green", no_wrap=True)
        table.add_column("End", style="green", no_wrap=True)
        table.add_column("Text", style="white")

        if segments:
            for segment in segments:
                text_cell = Text(segment.get("base_text", ""), style="white")
                highlight_text = segment.get("highlight_text", "")
                if highlight_text:
                    text_cell.append(highlight_text, style="bold yellow")

                table.add_row(
                    Text(_format_speaker(segment.get("speaker")), style="cyan"),
                    Text(_format_time_component(segment.get("start")), style="green"),
                    Text(_format_time_component(segment.get("end")), style="green"),
                    text_cell,
                )
        else:
            table.add_row(
                Text("--", style="dim"),
                Text("?", style="dim"),
                Text("?", style="dim"),
                Text("Waiting for transcription...", style="italic dim"),
            )

        renderables = [header, table]

        if buffer_msg:
            renderables.append(Text(f"Buffer: {buffer_msg}", style="cyan italic"))

        if info_msg:
            renderables.append(Text(info_msg, style="yellow"))

        return Group(*renderables)

    try:
        with Live(
            _build_renderable(
                display_segments_state, status_display, buffer_text, info_message
            ),
            console=console,
            refresh_per_second=12,
            vertical_overflow="visible",
        ) as live:
            async for update in results:
                info_message = ""
                data = update.to_dict() if hasattr(update, "to_dict") else update

                try:
                    logger.info(json.dumps(data, ensure_ascii=False))
                except Exception:  # pragma: no cover - logging should not break flow
                    pass

                status_value = data.get("status")
                if status_value:
                    status_display = status_value

                lines = data.get("lines") or []
                new_segments = _build_segments(lines)

                if not new_segments and segments_state:
                    new_segments = [segment.copy() for segment in segments_state]

                new_signature = [_segment_signature(segment) for segment in new_segments]
                display_segments: list[dict] = []

                buffer_text = (data.get("buffer_transcription") or "").strip()

                if not lines and not buffer_text and not status_value:
                    info_message = json.dumps(data, ensure_ascii=False)

                for idx, segment in enumerate(new_segments):
                    text_value = segment.get("text", "")
                    previous_text = ""
                    if idx < len(last_render_signature):
                        previous_text = last_render_signature[idx][3]

                    highlight_text = ""
                    if text_value != previous_text:
                        if previous_text and text_value.startswith(previous_text):
                            highlight_text = text_value[len(previous_text) :]
                        else:
                            highlight_text = text_value

                    base_length = len(text_value) - len(highlight_text)
                    base_text = text_value[:base_length]

                    display_segments.append(
                        {
                            "speaker": segment.get("speaker"),
                            "start": segment.get("start"),
                            "end": segment.get("end"),
                            "base_text": base_text,
                            "highlight_text": highlight_text,
                        }
                    )

                segments_state = [segment.copy() for segment in new_segments]
                display_segments_state = [segment.copy() for segment in display_segments]
                last_render_signature = new_signature

                live.update(
                    _build_renderable(
                        display_segments_state,
                        status_display,
                        buffer_text,
                        info_message,
                    )
                )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - guard the UI loop
        console.print(f"[WARN] result consumer stopped: {exc}")


async def run_processor() -> tuple[AudioProcessor, AsyncGenerator]:
    """Initialise the transcription engine and return processor + result stream."""
    engine = TranscriptionEngine(
        model_size=DEFAULT_MODEL_SIZE,
        lan=DEFAULT_LANGUAGE,
        backend=DEFAULT_BACKEND,
        diarization=DEFAULT_ENABLE_DIARIZATION,
        vac=DEFAULT_ENABLE_VAC,
        pcm_input=True,
        min_chunk_size=DEFAULT_MIN_CHUNK_SIZE,
    )
    processor = AudioProcessor(transcription_engine=engine)
    results = await processor.create_tasks()
    return processor, results


async def stream_microphone(device: str | None = None) -> None:
    """Capture microphone audio via FFmpeg and transcribe in-process."""
    processor, results = await run_processor()
    consumer = asyncio.create_task(recv_results(results))

    cmd = ffmpeg_cmd_for_mic(device)
    console.print("[INFO] Launching FFmpeg:", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_task = asyncio.create_task(_drain_stderr(proc, "mic"))

    try:
        await stream_pcm_to_processor(proc, processor)
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                proc.kill()
        await stderr_task
        await consumer
        await processor.cleanup()


async def stream_file(path: str) -> None:
    """Transcode a file with FFmpeg and transcribe in-process."""
    processor, results = await run_processor()
    consumer = asyncio.create_task(recv_results(results))

    cmd = ffmpeg_cmd_for_file(path)
    console.print("[INFO] Launching FFmpeg:", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_task = asyncio.create_task(_drain_stderr(proc, "file"))

    try:
        await stream_pcm_to_processor(proc, processor)
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                proc.kill()
        await stderr_task
        await consumer
        await processor.cleanup()


###############################################################################
# CLI
###############################################################################
def _usage() -> None:
    print(
        "Usage:\n"
        "  python whisperlivekit_inprocess_client.py mic [avfoundation_device]\n"
        "  python whisperlivekit_inprocess_client.py file <path-to-audio>\n"
        "\n"
        "Environment:\n"
        "  WLK_BACKEND             Backend name (default simulstreaming)\n"
        "  WLK_MODEL_SIZE          Whisper model size (default base)\n"
        "  WLK_LANGUAGE            Language hint (default en)\n"
        "  WLK_DIARIZATION         Enable diarization (default false)\n"
        "  WLK_VAC                 Enable voice activity control (default true)\n"
        "  WLK_MIN_CHUNK_SECONDS   Minimum chunk size for processor\n"
        "  WLK_CHUNK_SECONDS       PCM chunk size sent to processor (default 0.5s)\n"
        "  WLK_MIC_DEVICE          FFmpeg microphone device (default :0 for macOS)\n"
        "  WLK_MIC_FORMAT          FFmpeg input format (default avfoundation)\n"
        "  WLK_SAMPLE_RATE         Sample rate for PCM capture (default 16000)\n"
        "  WLK_FILE_CHUNK_SECONDS  Use -re pacing in FFmpeg for files (default off)\n"
        "\n"
        "Examples:\n"
        "  python whisperlivekit_inprocess_client.py mic :0\n"
        "  python whisperlivekit_inprocess_client.py file sample.wav\n"
    )


async def _amain(argv: Iterable[str]) -> None:
    argv = list(argv)
    if len(argv) < 2:
        _usage()
        return

    mode = argv[1]

    if mode == "mic":
        device = argv[2] if len(argv) >= 3 else DEFAULT_MIC_DEVICE
        await stream_microphone(device)
    elif mode == "file":
        if len(argv) < 3:
            _usage()
            return
        path = argv[2]
        if not os.path.exists(path):
            print(f"[ERR] File not found: {path}")
            return
        await stream_file(path)
    else:
        _usage()


def _main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass
    try:
        loop.run_until_complete(_amain(sys.argv))
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    _main()
