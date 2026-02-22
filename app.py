import onnxruntime as ort
import librosa
import numpy as np
import gradio as gr
import os
from typing import Tuple, Optional

# Load model
model = ort.InferenceSession("infant_cry.onnx", providers=['CPUExecutionProvider'])
CLASSES = ['Hungry 🍼', 'Discomfort 🏗️', 'Tired 😴', 'Pain 😫']
TARGET_LENGTH = 300 

def preprocess_audio(audio_input) -> Optional[np.ndarray]:
    try:
        if isinstance(audio_input, tuple):
            sr, samples = audio_input
            audio = samples.astype(np.float32)
        else:
            audio, sr = librosa.load(audio_input, sr=None)
        
        # 1. Normalization (Volume fix)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # 2. Convert to mono and resample
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # 3. MFCC Extraction
        mfcc = librosa.feature.mfcc(
            y=audio, sr=16000, n_mfcc=40, n_fft=1024, hop_length=160
        ).T
        
        # 4. Target Length check
        if mfcc.shape[0] < TARGET_LENGTH:
            mfcc = np.pad(mfcc, ((0, TARGET_LENGTH - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:TARGET_LENGTH]
            
        return mfcc[np.newaxis,...].astype(np.float32)
        
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        return None

def predict(audio_input):
    features = preprocess_audio(audio_input)
    if features is None:
        return "Invalid audio"
    
    outputs = model.run(None, {'input': features})[0][0]
    
    # Isse results bar graph ke roop mein dikhenge
    return {CLASSES[i]: float(outputs[i]) for i in range(len(CLASSES))}

# UI with Label instead of JSON
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👶 Baby Cry Language Translator")
    gr.Markdown("Identify why the baby is crying using AI.")
    
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(label="Record or Upload Cry", type="filepath")
            btn = gr.Button("Analyze Voice", variant="primary")
        with gr.Column():
            # Label component results ko sundar dikhata hai
            output_label = gr.Label(label="Analysis Result", num_top_classes=4)

    btn.click(predict, inputs=[audio_in], outputs=[output_label])

if __name__ == "__main__":
    # Render port handling
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
