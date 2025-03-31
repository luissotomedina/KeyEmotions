"""
keyEmotionsUI.py

KeyEmotions UI for selecting emotions and generating music.
"""
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QGridLayout, 
                            QLabel, QPushButton, QFrame)

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
        self.initUI()

    def initUI(self):
        """
        Initialize the user interface for the EmotionGridSelector.
        """
        self.setWindowTitle('Key Emotions Selector')
        self.setGeometry(100, 100, 400, 400)
        
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
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        title_label = QLabel("KeyEmotions\nSelect dominant emotion")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-family: Arial;
            }
        """)
        
        title_font = title_label.font()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        
        title_label.setText("""
            <div style='font-size: 24pt; font-weight: bold; color: #2b2d42;'>KeyEmotions</div>
            <div style='font-size: 12pt; color: #6c757d; margin-top: 5px;'>Select dominant emotion</div>
        """)
        
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

    def generate_music(self):
        """
        Generate music with selected emotion
        """
        if self.selected_quadrant is None:
            return
            
        emotion = self.emotions[self.selected_quadrant]
        print(f"\nGenerating music with emotion: {emotion['name']}")
        
        self.generate_btn.setText("Generating... 🎶")
        self.generate_btn.setEnabled(False)
        QApplication.processEvents()
        
        generator = KeyEmotionsGenerator()
        generator.generate_and_save(emotion['idx'], 19, output_path="output")
        # run_KeyEmotions(emotion['idx'], 19, output_path="output")

        self.generate_btn.setText("Generate Music")
        self.generate_btn.setEnabled(True)
        print("Music generated successfully!\n")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionGridSelector()
    window.show()
    sys.exit(app.exec())