import tempfile
import subprocess
from typing import Optional
from glob import glob

import torchaudio


from daisy.abstract import AudioDownloader, AudioItem, DownloadItem


class VideoAudioDownloader(AudioDownloader):
    def __init__(
        self,
        save_dir: str,
        overwrite: bool = False,
        section_length: Optional[int] = 120,
        max_workers: Optional[int] = None,
    ):
        super().__init__(
            save_dir, overwrite, ["youtube.com", "bilibili.com"], max_workers
        )
        self.section_length = section_length

    def download(self, item: AudioItem) -> DownloadItem:
        with tempfile.TemporaryDirectory() as temp_dir:
            if item.parsed_duration is None:
                item.parsed_duration = int(item.duration)
            command = [
                "yt-dlp",
                # "-f",
                # "(bestaudio)[protocol!*=dash]",
                "--external-downloader",
                "ffmpeg",
                "--output",
                f"{temp_dir}/audio.%(ext)s",
                "--extract-audio",
                "--audio-quality",
                "0",
                "--limit-rate",
                "500K",
                "--retries",
                "10",
                "--fragment-retries",
                "10",
                "--skip-unavailable-fragments",
                "--user-agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                # "--quiet",
                item.url,
                "--cookies",
                "cookies.txt",
                "--downloader-args",
                "ffmpeg_i:-http_persistent 0",
            ]
            if (
                self.section_length is not None
                and item.parsed_duration > self.section_length
            ):
                video_length = item.parsed_duration
                if video_length == -1:
                    raise ValueError("Shorts are not supported with section_length")
                midpoint = video_length / 2
                start = int(midpoint - self.section_length / 2)
                end = int(midpoint + self.section_length / 2)
                command.extend(
                    [
                        "--download-sections",
                        f"*{start}-{end}",
                    ]
                )
                segment = (start, end)
            else:
                segment = None
            print("running command", command)
            try:
                subprocess.run(
                    command,
                    check=True,
                )
                audio_path = glob(f"{temp_dir}/audio.*")[0]
                audio_format = audio_path.split(".")[-1]
            except subprocess.CalledProcessError as e:
                print(f"Error downloading audio: {e}")
                return None
            audio, sr = torchaudio.load(f"{temp_dir}/audio.{audio_format}")
            audio = audio.numpy()[0]
            return DownloadItem(
                sr=sr,
                audio=audio,
                segment=segment,
                media_item_id=item.media_item_id,
                audio_item_id=item.identifier,
                identifier=item.identifier,
            )
