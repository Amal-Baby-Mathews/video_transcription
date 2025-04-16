import json
import os
from ollama import chat

# --- Configuration ---
model_name = "gemma3:1b"  # Or your preferred Ollama model
transcript_path = "transcript_temp.txt"
output_path = "client_requirements_extracted.txt"
segment_length = 500  # Approx characters per segment. Adjust as needed.
# Define the specific information you want to extract
information_topic = "Client Requirements regarding AI contributions to their existing project. Specifically, what tasks or functionalities they need the AI team to implement or assist with."
# --- End Configuration ---

# Step 1: Verify Ollama server and model
print(f"Checking Ollama server and model '{model_name}'...")
try:
    # Send a simple message to test connection and model availability
    test_response = chat(model=model_name, messages=[
        {'role': 'user', 'content': 'Hello!'}
    ])
    print("Ollama server connected and model is available.")
except Exception as e:
    print(f"Error connecting to Ollama or accessing model '{model_name}': {e}")
    print("Please ensure the Ollama server is running ('ollama serve')")
    print(f"And that the model '{model_name}' is installed ('ollama pull {model_name}').")
    exit(1)

# Step 2: Load transcript
if not os.path.exists(transcript_path):
    print(f"Error: Transcript file '{transcript_path}' not found.")
    print("Please create the transcript file or update the 'transcript_path' variable.")
    exit(1)

print(f"Loading transcript from '{transcript_path}'...")
try:
    with open(transcript_path, "r", encoding="utf-8") as f: # Added encoding
        transcript_text = f.read()
except Exception as e:
    print(f"Error reading transcript file: {e}")
    exit(1)

if not transcript_text.strip():
    print("Error: Transcript file is empty.")
    exit(1)

# Step 3: Segment the transcript (Simple character-based segmentation)
print(f"Segmenting transcript into chunks of approximately {segment_length} characters...")
segments = []
start_char = 0
while start_char < len(transcript_text):
    end_char = min(start_char + segment_length, len(transcript_text))
    # Try to break at a sentence boundary (heuristic)
    potential_end = transcript_text.rfind('.', start_char, end_char)
    if potential_end > start_char + (segment_length / 2): # Avoid tiny segments
         end_char = potential_end + 1

    segment_text = transcript_text[start_char:end_char]
    if segment_text.strip(): # Avoid adding empty segments
        segments.append({
            "start_char": start_char,
            "end_char": end_char,
            "text": segment_text.strip()
            # Future enhancement: Add speaker info if available (e.g., from VTT)
            # "speaker": segment.get("speaker", "Unknown")
        })
    start_char = end_char

print(f"Transcript divided into {len(segments)} segments.")

# Step 4: Initialize output file
print(f"Initializing output file '{output_path}'...")
try:
    with open(output_path, "w", encoding="utf-8") as f: # Added encoding
        f.write(f"Extracted Information: {information_topic}\n")
        f.write(f"Source Transcript: {transcript_path}\n")
        f.write("=" * 60 + "\n\n")
except Exception as e:
    print(f"Error creating or writing to output file '{output_path}': {e}")
    exit(1)

# Step 5: Process each segment with Ollama using the structured prompt
print("Starting extraction process...")
total_extracted_count = 0

for i, segment in enumerate(segments):
    segment_text = segment["text"]
    start_char = segment["start_char"]
    end_char = segment["end_char"]
    # speaker = segment.get("speaker", "Unknown") # Use when speaker info is added

    print(f"\nProcessing segment {i+1}/{len(segments)} (Chars {start_char}-{end_char})...")

    # --- Construct the structured prompt ---
    structured_prompt = f"""
**Role:** You are an AI assistant specialized in accurately extracting specific information from text segments, particularly transcripts of meetings or workshops.
**Objective:** Identify and extract explicit statements related to the defined **Information Topic** from the provided **Transcript Segment**.
**Input:**
*   **Information Topic:** {information_topic}
*   **Transcript Segment:**
    ```
    {segment_text}
    ```

**Instructions:**
1.  **Focus:** Concentrate *only* on information directly relevant to the specified **Information Topic**.
2.  **Accuracy:** Extract *only* what is explicitly stated in the text. Do not assume, or add external information.
3.  **Format:**
    *   If relevant information is found, list each distinct point as a numbered item.
    *   Phrase each item concisely, capturing the core statement. Avoid verbatim quotes unless absolutely necessary for meaning.
4.  **No Information:** If *no* statements directly related to the **Information Topic** are present in the segment, output the single word: `NONE`.

**Output:**
[Your extracted numbered list or the word NONE]
"""
    # --- End of structured prompt ---

    try:
        # Call Ollama API
        response = chat(model=model_name, messages=[
            # Minimal system prompt (optional, can sometimes help focus the model)
            # {'role': 'system', 'content': 'Follow the user\'s instructions precisely.'},
            {'role': 'user', 'content': structured_prompt}
            # Removed the non-standard 'extractor' role
        ])
        extracted_info = response['message']['content'].strip()

        # Process and save the result
        if extracted_info.upper() != "NONE":
            total_extracted_count += 1
            print(f"  -> Found relevant information in segment {i+1}.")
            # Include segment context in the output file
            output_block = f"--- Segment {i+1} (Chars {start_char} - {end_char}) ---\n"
            # output_block += f"Speaker: {speaker}\n" # Add when speaker info is available
            output_block += f"Extracted Information:\n{extracted_info}\n"
            output_block += "-" * 20 + "\n\n"

            with open(output_path, "a", encoding="utf-8") as f: # Added encoding
                f.write(output_block)
        else:
            print(f"  -> No relevant information found in segment {i+1}.")

    except Exception as e:
        print(f"  -> Error processing segment {i+1} with Ollama: {e}")
        # Optionally write error to file or handle differently
        with open(output_path, "a", encoding="utf-8") as f:
             f.write(f"--- ERROR processing Segment {i+1} (Chars {start_char} - {end_char}) ---\n")
             f.write(f"Error: {e}\n")
             f.write("-" * 20 + "\n\n")

# Step 6: Finalize
print("\n" + "=" * 60)
print("Extraction process complete.")
print(f"Found relevant information in {total_extracted_count} out of {len(segments)} segments.")
print(f"Results saved to '{output_path}'")
print("=" * 60)