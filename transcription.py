import subprocess
import whisper
from pyannote.audio import Pipeline
import re
import os

# Step 1: Extract audio from video using FFmpeg
video_path = "/home/seq_amal/work_temp/video_transcription/video_transcription/AI _ Workshop 1 _ _ anonymous _ Microsoft Teams 2025-04-15 17-31-49.mp4"
audio_path = "presentation.wav"
subprocess.run([
    "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
    "-ar", "16000", "-ac", "1", audio_path
])

# Step 2: Transcribe with Whisper (small model)
model = whisper.load_model("small", device="cpu")  # Use small model, force CPU
result = model.transcribe(audio_path, language="en", fp16=False)  # Explicit FP32
transcript = result["segments"]

# Print each transcribed segment
print("Transcribing segments...")
for segment in transcript:
    line = f"[{segment['start']:.3f} --> {segment['end']:.3f}] {segment['text'].strip()}"
    print(line)

# Step 3: Speaker diarization with pyannote.audio
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline(audio_path)

# Step 4: Combine transcription and diarization
output = []
print("\nCombining with speaker diarization...")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start = turn.start
    end = turn.end
    segment_text = ""
    for segment in transcript:
        if segment["start"] >= start and segment["end"] <= end:
            segment_text += segment["text"] + " "
    if segment_text.strip():
        line = f"[{start:.3f} --> {end:.3f}] {speaker}: {segment_text.strip()}"
        output.append(line)
        print(line)

# Step 5: Save clean transcript
with open("transcript.txt", "w") as f:
    f.write("\n".join(output))

# Step 6: Clean up temporary files
os.remove(audio_path)
print("\nTranscription complete. Output saved to transcript.txt")