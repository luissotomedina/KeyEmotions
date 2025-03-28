# KeyEmotions

KeyEmotions is a music generation system that combines emotional expression with machine learning. It leverages a Transformer-based model to generate music compositions based on user-selected emotional quadrants. The project utilizes MIDI data for training the model and generates compositions that reflect specific emotions.

This project is the **final thesis for a Master's degree in Artificial Intelligence**, created as part of the academic requirements to complete the degree. The system aims to explore the intersection of emotional expression and artificial intelligence in music creation.

## Features

- **Emotion-based Music Generation**: Select an emotion from a predefined set (e.g., Angry, Excited, Sad, Calm) to generate music based on that emotion.
- **Interactive UI**: A simple, clean, and intuitive graphical user interface (GUI) built using PyQt6 to interact with the model and generate music.
- **MIDI Processing**: Use MIDI files as the base data for training and generating compositions.
- **Model**: Transformer-based autoregressive model for music generation.

## Demo

You can see a live demo of the application at the following link:  
[https://luissotomedina.github.io/portfolio/KeyEmotions/](https://luissotomedina.github.io/portfolio/KeyEmotions/)

## Prerequisites

### Dependencies

To run this project, you need to install the following dependencies:

- Python 3.12+
- PyQt6
- Torch (with CUDA support, if available)
- PyYAML
- Mido (for MIDI processing)
- Fluidsynth (for MIDI sound synthesis)
- Pandas, Seaborn, Matplotlib, Numpy (for data processing and visualization)

You can install the dependencies by running:

```bash
pip install -r requirements.txt
