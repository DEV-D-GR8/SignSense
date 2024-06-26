# SignSense: Transformer based ASL Recognition Model

This repository contains a transformer-based model for real-time American Sign Language (ASL) recognition. The model leverages state-of-the-art transformer architecture to accurately interpret ASL gestures and utilizes the Gemini-Pro LLM API for constructing sentences from recognized ASL signs. The system supports live video input for seamless, on-the-fly translation of ASL gestures into textual form, aiding communication accessibility.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup and Testing](#setup-and-testing)
- [Demo](#demo)
- [License](#license)

## Project Overview
This project aims to provide a robust solution for real-time ASL recognition using a transformer-based deep learning model. The model captures live video input, processes the frames to detect and recognize ASL gestures, and constructs meaningful sentences from the recognized words using the Gemini-Pro LLM API. This tool can significantly enhance communication for individuals who use ASL.

## Features
- **Real-time ASL Recognition**: Detect and recognize ASL gestures in real-time.
- **Transformer Architecture**: Utilizes advanced transformer models for high accuracy.
- **Sentence Construction**: Integrates with Gemini-Pro LLM API to build sentences from recognized signs.
- **Live Video Input**: Supports live video input for seamless ASL translation.

## Tech Stack
- **Python**: Core programming language used for development.
- **TensorFlow**: Deep learning framework for building and training the transformer model.
- **OpenCV**: Library for real-time computer vision tasks, used for video capture and preprocessing.
- **MediaPipe**: Framework for building multimodal machine learning pipelines, used for hand and gesture tracking.
- **Gemini-Pro LLM API**: API for generating sentences from recognized ASL words.

## Setup and Testing

To run the project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/DEV-D-GR8/SignSense.git
   cd SignSense

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the Application:**
   ```bash
   python app.py

## Demo
For a visual demonstration, check out my [YouTube video](https://youtu.be/6XNY6YBXgyI?si=RoZdn_8jL35EMuYD).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more information.
