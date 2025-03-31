# KeyEmotions  

**KeyEmotions** is a music generation system that combines emotional expression with machine learning. It uses a Transformer-based model to generate music compositions based on user-selected emotional quadrants. The project processes MIDI data for training and produces emotionally expressive musical outputs.  

This project was developed as the **final thesis for a Master's degree in Artificial Intelligence**, fulfilling academic requirements for the degree.  

---

## 🎹 Features  
- **Emotion-based Music Generation** – Generate music by selecting emotions (e.g., Angry, Excited, Sad, Calm).  
- **Interactive UI** – User-friendly PyQt6 interface for model interaction.  
- **MIDI Processing** – Uses MIDI files for training and generation.  
- **Transformer Model** – Autoregressive architecture for music generation.  

---

## 🎥 Demo  
Check out the demo here:  
🔗 [https://luissotomedina.github.io/portfolio/KeyEmotions/](https://luissotomedina.github.io/portfolio/KeyEmotions/)  

---

## ⚙️ Installation  

### 📋 Prerequisites  
- **Python 3.12+**  
- **Dependencies**: PyQt6, PyTorch (CUDA optional), Mido, Fluidsynth, Pandas, Matplotlib, etc.  

Install dependencies with:  
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 🔄 Preprocessing
Place MIDI files in `data/raw/`.

Run preprocessing:

```bash
python src/preprocessing/run_preprocessing.py
```

### 🏗️ Dataset Creation
To generate the training dataset:

```bash
python src/preprocessing/loader.py
```

### 🎓 Training
Configure training parameters in `src/config/default.yaml`, then run:

```bash
python src/train.py
```

### 🎵 Generating Music
Place `config.json` and `weights.pt` in the same directory.

Launch the UI and specify model paths + output location:

```bash
python src/KeyEmotionsUI.py
```


