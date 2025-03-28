import os

folders = [
    "data/raw",
    "data/processed",    
    "data/prepared",
    "data/plots",
    "experiments",
    "outputs"
]

def create_folders():
    print("Configuring folders...")

    for folder in folders: 
        os.makedirs(folder, exist_ok=True)

    print("Folders configured successfully!")

if __name__ == "__main__":
    create_folders()