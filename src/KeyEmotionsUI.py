"""
keyEmotionsUI.py

KeyEmotions UI for selecting emotions and generating music.
"""
import sys
import os
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QGridLayout, 
                            QLabel, QPushButton, QFrame, QFileDialog, QHBoxLayout)

from generate import *

class EmotionGridSelector(QWidget):
    """
    EmotionGridSelector is a PyQt6 widget that allows users to select an emotion
    from a grid of options. Each option is represented by a frame containing an emoji
    and a label. The selected emotion is highlighted, and a button is provided to
    generate music based on the selected emotion.
    """
    def __init__(self):
        """
        Initialize the EmotionGridSelector widget.
        """
        super().__init__()
        self.selected_quadrant = None
        self.output_path = str(Path.home() / "Desktop")
        self.model_weights_path = str(Path(__file__).parent.parent / "experiments")
        self.initUI()

    def initUI(self):
        """
        Initialize the user interface for the EmotionGridSelector.
        """
        self.setWindowTitle('Key Emotions Selector')
        self.setGeometry(100, 100, 700, 600)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                font-family: Arial;
            }
            QFrame {
                border: 2px solid #dee2e6;
                border-radius: 8px;
                background-color: white;
            }
            QFrame:hover {
                border: 2px solid #adb5bd;
            }
            QLabel {
                font-size: 18px;
                color: #212529;
            }
            QFrame[accessibleName="emotionButton"] QLabel {
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
                border-radius: 4px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:disabled {
                background-color: #d6d8db;
                color: #6c757d;
            }
            #pathLabel {
                font-size: 12px;
                color: #495057;
                background-color: #e9ecef;
                padding: 5px;
                border-radius: 4px;
            }
            .settingsButton {
                background-color: #4a6fa5;
                font-size: 12px;
            }
            .settingsButton:hover {
                background-color: #3a5a80;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title_label = QLabel("""
            <div style='font-size: 24pt; font-weight: bold; color: #2b2d42;'>KeyEmotions</div>
            <div style='font-size: 12pt; color: #6c757d; margin-top: 5px;'>Select dominant emotion</div>
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        grid_layout.setContentsMargins(10, 10, 10, 10)

        self.emotions = {
            2: {"name": "Angry",    "emoji": "😡", "idx": 2, "border_color": "#d23f2f", "bck_color": "#e68989", "row": 0, "col": 0},
            1: {"name": "Excited",  "emoji": "🎉", "idx": 1, "border_color": "#ffe70a", "bck_color": "#fff599", "row": 0, "col": 1},
            3: {"name": "Sad",      "emoji": "😢", "idx": 3, "border_color": "#b0bec5", "bck_color": "#dce2e5", "row": 1, "col": 0},
            4: {"name": "Calm",     "emoji": "😊", "idx": 4, "border_color": "#64b5f6", "bck_color": "#b2dafb", "row": 1, "col": 1}
        }

        self.emotion_frames = {}
        for emotion_id, props in self.emotions.items():
            frame = QFrame()
            frame.setAccessibleName("emotionButton")
            frame.setCursor(Qt.CursorShape.PointingHandCursor)
            frame.mousePressEvent = lambda event, eid=emotion_id: self.select_emotion(eid)
            
            emotion_layout = QVBoxLayout()
            emotion_layout.setContentsMargins(10, 10, 10, 10)
            
            emotion_label = QLabel(f"{props['emoji']} {props['name']}")
            emotion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            emotion_label.setFont(QFont("Arial", 14))
            
            emotion_layout.addWidget(emotion_label)
            frame.setLayout(emotion_layout)
            
            grid_layout.addWidget(frame, props["row"], props["col"])
            self.emotion_frames[emotion_id] = frame

        main_layout.addLayout(grid_layout)

        output_path_layout = QHBoxLayout()
        output_path_layout.setSpacing(10)
        
        self.output_path_display = QLabel(self.output_path)
        self.output_path_display.setObjectName("pathLabel")
        self.output_path_display.setWordWrap(True)
        self.output_path_display.setFixedHeight(40)
        self.output_path_display.setMinimumWidth(250)
        self.output_path_display.setStyleSheet("""
            QLabel {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
        """)
        
        output_path_button = QPushButton("Select Output Folder")
        output_path_button.setFixedSize(150, 40)
        output_path_button.setFont(QFont("Arial", 10))
        output_path_button.setProperty("class", "settingsButton")
        output_path_button.clicked.connect(self.select_output_path)
        
        output_path_layout.addWidget(self.output_path_display)
        output_path_layout.addWidget(output_path_button)
        
        main_layout.addLayout(output_path_layout)

        model_path_layout = QHBoxLayout()
        model_path_layout.setSpacing(10)
        
        self.model_path_display = QLabel(self.model_weights_path)
        self.model_path_display.setObjectName("pathLabel")
        self.model_path_display.setWordWrap(True)
        self.model_path_display.setFixedHeight(40) 
        self.model_path_display.setMinimumWidth(250)
        self.model_path_display.setStyleSheet("""
            QLabel {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
        """)
        
        model_path_button = QPushButton("Locate Model Weights")
        model_path_button.setFixedSize(150, 40)
        model_path_button.setFont(QFont("Arial", 10))
        model_path_button.setProperty("class", "settingsButton")
        model_path_button.clicked.connect(self.select_model_weights_path)
        
        model_path_layout.addWidget(self.model_path_display)
        model_path_layout.addWidget(model_path_button)
        
        main_layout.addLayout(model_path_layout)

        # Generate button
        self.generate_btn = QPushButton("Generate Music 🎵")
        self.generate_btn.setFont(QFont("Arial", 12))
        self.generate_btn.clicked.connect(self.generate_music)
        self.generate_btn.setEnabled(False)
        main_layout.addWidget(self.generate_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.setLayout(main_layout)

    def select_emotion(self, emotion_id):
        """
        Select emotion apply background color

        Parameters:
            emotion_id (int): ID of the selected emotion.
        """
        for eid, frame in self.emotion_frames.items():
            frame.setStyleSheet("QFrame { background-color: white; border: 2px solid #dee2e6; }")
        
        border_color = self.emotions[emotion_id]["border_color"]
        bck_color = self.emotions[emotion_id]["bck_color"]
        self.emotion_frames[emotion_id].setStyleSheet(
            f"QFrame {{ background-color: {bck_color}; border: 2px solid {border_color}; }}"
        )
        
        self.selected_quadrant = emotion_id
        self.generate_btn.setEnabled(True)
        print(f"Selected emotion: {self.emotions[emotion_id]['name']}")

    def select_output_path(self):
        """
        Open a dialog to select the output directory.
        """
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_path,
            QFileDialog.Option.ShowDirsOnly
        )
        
        if path:
            self.output_path = path
            self.output_path_display.setText(path)
            print(f"Output path set to: {path}")

    def select_model_weights_path(self):
        """
        Open a dialog to select the model weights directory.
        """
        path = QFileDialog.getExistingDirectory(
            self,
            "Locate Model Weights Directory",
            self.model_weights_path,
            QFileDialog.Option.ShowDirsOnly
        )
        
        if path:
            self.model_weights_path = path
            self.model_path_display.setText(path)
            print(f"Model weights path set to: {path}")

    def generate_music(self):
        """
        Generate music with selected emotion.
        """
        if self.selected_quadrant is None:
            return
            
        emotion = self.emotions[self.selected_quadrant]
        print(f"\nGenerating music with emotion: {emotion['name']}")
        print(f"Output directory: {self.output_path}")
        print(f"Model weights directory: {self.model_weights_path}")
        
        self.generate_btn.setText("Generating... 🎶")
        self.generate_btn.setEnabled(False)
        QApplication.processEvents()
        
        try:
            os.makedirs(self.output_path, exist_ok=True)
            
            generator = KeyEmotionsGenerator()
            generator.generate_and_save(
                emotion=emotion['idx'],
                output_path=self.output_path,
                exp_dir=self.model_weights_path
            )
            
            print("Music generated successfully!\n")
        except Exception as e:
            print(f"Error generating music: {str(e)}")
        finally:
            self.generate_btn.setText("Generate Music 🎵")
            self.generate_btn.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionGridSelector()
    window.show()
    sys.exit(app.exec())