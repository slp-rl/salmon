import os
import azure.cognitiveservices.speech as speechsdk
from utils.audio import Audio

"""
    voices - https://learn.microsoft.com/en-gb/azure/ai-services/speech-service/language-support?tabs=tts
    ssml - https://learn.microsoft.com/en-gb/azure/ai-services/speech-service/speech-synthesis-markup-voice#use-speaking-styles-and-roles
"""

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region='eastus')
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

# The neural multilingual voice can speak different languages based on the input text.

speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

speakers = [
    {
        "voice": "en-GB-SoniaNeural",
        "styles": ["cheerful", "sad"]
    },
    {
        "voice": "en-US-AriaNeural",
        "styles": ["angry", "chat", "cheerful", "customerservice", "empathetic", "excited", "friendly", "hopeful",
                   "narration-professional", "newscast-casual", "newscast-formal", "sad", "shouting", "terrified",
                   "unfriendly", "whispering"]
    },
    {
        "voice": "en-US-DavisNeural",
        "styles": ["angry", "cheerful", "excited", "friendly", "hopeful", "newscast", "sad", "shouting", "terrified",
                   "unfriendly",
                   "whispering"]
    },

    {
        "voice": "en-US-GuyNeural",
        "styles": ["angry", "cheerful", "excited", "friendly", "hopeful", "newscast", "sad", "shouting", "terrified",
                   "unfriendly",
                   "whispering"]
    },

    {
        "voice": "en-US-JaneNeural",
        "styles": ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly",
                   "whispering"]
    },
    {
        "voice": "en-US-JasonNeural",
        "styles": ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly",
                   "whispering"]
    },
    {
        "voice": "en-US-JennyNeural",
        "styles": ["angry", "assistant", "chat", "cheerful", "customerservice", "excited", "friendly", "hopeful",
                   "newscast", "sad", "shouting", "terrified", "unfriendly", "whispering"]
    },
    {
        "voice": "en-US-NancyNeural",
        "styles": ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly",
                   "whispering"]
    },
    {
        "voice": "en-US-SaraNeural",
        "styles": ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly",
                   "whispering"]
    },
    {
        "voice": "en-US-TonyNeural",
        "styles": ["angry", "cheerful", "excited", "friendly", "hopeful", "sad", "shouting", "terrified", "unfriendly",
                   "whispering"]
    }
]

styles = {
    "sad": "sad",
    "neutral": "neutral",
    "happy": "cheerful",
    "excited": "excited"
}


def generate_speech(out_path, text, speaker, style="neutral", style_degree=1.0):
    ssml = f"""<speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts'>
        <voice name='{speaker}'>
            <mstts:express-as style="{styles[style]}" styledegree="{style_degree}">
                {text}
            </mstts:express-as>
        </voice>
    </speak>"""
    speech_synthesis_result = speech_synthesizer.speak_ssml_async(ssml).get()
    if speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        raise Exception(f"Could not tts {speaker} with style {style}")
    stream = speechsdk.AudioDataStream(speech_synthesis_result)

    if out_path == "wav":
        stream.save_to_wav_file("tmp.wav")
        a = Audio("tmp.wav")
        os.remove("tmp.wav")
        return a
    else:
        stream.save_to_wav_file(out_path)
