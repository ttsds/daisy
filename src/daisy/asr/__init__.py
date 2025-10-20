from faster_whisper import WhisperModel
import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
import librosa
import numpy as np

from daisy.core.base import ASRModel
from daisy.core import LANGUAGES

class WhisperASRModel(ASRModel):
    def __init__(self, device: str = "cpu"):
        super().__init__(model_id="whisper_turbo")
        self.model = WhisperModel("turbo", device=device)
        self.sampling_rate = self.model.feature_extractor.sampling_rate

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
        segments, info = self.model.transcribe(audio)
        return " ".join([segment.text for segment in segments])
    
    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported language codes for Whisper.
        
        Returns:
            List of ISO 639-1 language codes supported by Whisper
        """
        # Whisper supports 99 languages with ISO 639-1 codes
        iso3_codes = [
            "eng",  # English
            "zho",  # Chinese
            "deu",  # German
            "spa",  # Spanish
            "rus",  # Russian
            "kor",  # Korean
            "fra",  # French
            "jpn",  # Japanese
            "por",  # Portuguese
            "tur",  # Turkish
            "pol",  # Polish
            "cat",  # Catalan
            "nld",  # Dutch
            "ara",  # Arabic
            "swe",  # Swedish
            "ita",  # Italian
            "ind",  # Indonesian
            "hin",  # Hindi
            "fin",  # Finnish
            "vie",  # Vietnamese
            "heb",  # Hebrew
            "ukr",  # Ukrainian
            "ell",  # Greek
            "msa",  # Malay
            "ces",  # Czech
            "ron",  # Romanian
            "dan",  # Danish
            "hun",  # Hungarian
            "tam",  # Tamil
            "nor",  # Norwegian
            "tha",  # Thai
            "urd",  # Urdu
            "hrv",  # Croatian
            "bul",  # Bulgarian
            "lit",  # Lithuanian
            "lat",  # Latin
            "mri",  # Māori
            "mal",  # Malayalam
            "cym",  # Welsh
            "slk",  # Slovak
            "tel",  # Telugu
            "fas",  # Persian
            "lav",  # Latvian
            "ben",  # Bengali
            "srp",  # Serbian
            "aze",  # Azerbaijani
            "slv",  # Slovenian
            "kan",  # Kannada
            "est",  # Estonian
            "mkd",  # Macedonian
            "bre",  # Breton
            "eus",  # Basque
            "isl",  # Icelandic
            "hye",  # Armenian
            "nep",  # Nepali
            "mon",  # Mongolian
            "bos",  # Bosnian
            "kaz",  # Kazakh
            "sqi",  # Albanian
            "swa",  # Swahili
            "glg",  # Galician
            "mar",  # Marathi
            "pan",  # Panjabi
            "sin",  # Sinhala
            "khm",  # Khmer
            "sna",  # Shona
            "yor",  # Yoruba
            "som",  # Somali
            "afr",  # Afrikaans
            "oci",  # Occitan
            "kat",  # Georgian
            "bel",  # Belarusian
            "tgk",  # Tajik
            "snd",  # Sindhi
            "guj",  # Gujarati
            "amh",  # Amharic
            "yid",  # Yiddish
            "lao",  # Lao
            "uzb",  # Uzbek
            "fao",  # Faroese
            "hat",  # Haitian
            "pus",  # Pashto
            "tuk",  # Turkmen
            "nno",  # Norwegian Nynorsk
            "mlt",  # Maltese
            "san",  # Sanskrit
            "ltz",  # Luxembourgish
            "mya",  # Burmese
            "bod",  # Tibetan
            "tgl",  # Tagalog
            "mlg",  # Malagasy
            "asm",  # Assamese
            "tat",  # Tatar
            "haw",  # Hawaiian
            "lin",  # Lingala
            "hau",  # Hausa
            "bak",  # Bashkir
            "jav",  # Javanese
            "sun",  # Sundanese
        ]
        return iso3_codes


class MMSASRModel(ASRModel):
    """
    Massively Multilingual Speech (MMS) ASR model from Facebook.
    Supports 1000+ languages with adapter-based language switching.
    """
    
    def __init__(self, device: str = "cpu", language: str = "en"):
        """
        Initialize the MMS ASR model.
        
        Args:
            device: Device to run the model on ("cpu" or "cuda")
            language: Target language code (ISO 639-2 format, e.g., "en", "fr", "es")
        """
        super().__init__(model_id="mms_1b_all")
        self.device = device

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(f"facebook/mms-1b-all")
        self.model = Wav2Vec2ForCTC.from_pretrained(f"facebook/mms-1b-all")
        
        # Set target language and load adapter
        language = LANGUAGES[language].iso3
        self.set_language(language)
    
    def set_language(self, language_code: str) -> None:
        """
        Switch the model to a different language.
        
        Args:
            language_code: ISO 639-3 language code (e.g., "eng", "fra", "spa")
        """
        self.target_language = language_code
        self.processor.tokenizer.set_target_lang(language_code)
        self.model.load_adapter(language_code)
    
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Process audio
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Move to device if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self.model = self.model.to(self.device)
        
        # Transcribe
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        
        # Decode
        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.processor.decode(ids)
        
        return transcription
    
    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of ISO 639-3 language codes
        """
        return list(self.processor.tokenizer.vocab.keys())