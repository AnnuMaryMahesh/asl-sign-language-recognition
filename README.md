# Real-Time Sign Language Recognition — KNN Classifier
### MML Project | Non-Parametric Techniques

---

## Files in this folder

| File | Purpose |
|---|---|
| `collect_data.py` | Step 1 — Collect hand sign data from webcam |
| `train_model.py` | Step 2 — Train the KNN model |
| `recognize.py` | Step 3 — Run real-time recognition |
| `requirements.txt` | All Python libraries needed |

---

## Setup (do this once)

### 1. Install Python
Download Python 3.10 or newer from https://python.org
During installation on Windows, tick **"Add Python to PATH"**

### 2. Install VS Code
Download from https://code.visualstudio.com
Install the **Python extension** inside VS Code

### 3. Open the project
- File → Open Folder → select this `sign_language_knn` folder

### 4. Open terminal in VS Code
Press **Ctrl + `** (backtick)

### 5. Create a virtual environment
```
python -m venv .venv
```

### 6. Activate it
```
# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
```
You should see `(.venv)` appear at the start of the terminal line.

### 7. Install all libraries
```
pip install -r requirements.txt
```
This will take a few minutes the first time.

---

## Running the project

### Step 1 — Collect data
```
python collect_data.py
```
- A webcam window opens
- **Click on the window** to give it focus
- Make the ASL hand sign shown on screen
- Press **SPACEBAR** to start collecting
- Hold the sign still for ~3 seconds (counts to 100)
- Repeat for all 26 letters A–Z
- Press **Q** to quit at any time

### Step 2 — Train the model
```
python train_model.py
```
- Trains the KNN classifier on your collected data
- Finds the best K value automatically
- Saves model to `data/knn_model.pkl`
- Prints accuracy and saves charts to `data/`

### Step 3 — Real-time recognition
```
python recognize.py
```
- Webcam opens — show any ASL sign
- Predicted letter shown with confidence bar
- Hold a sign for ~1 second to add it to the word

**Controls:**
| Key | Action |
|---|---|
| Q | Quit |
| B | Delete last letter |
| C | Clear word |
| Space | Add a space |

---

## Troubleshooting

**Camera shows black screen:**
Open `collect_data.py` and `recognize.py`, change `VideoCapture(0, ...)` to `VideoCapture(1, ...)` or `VideoCapture(2, ...)`

**mediapipe error on import:**
```
pip uninstall mediapipe -y
pip install mediapipe==0.10.9
```

**"No such file or directory" for train_model.py:**
Make sure you are inside the `sign_language_knn` folder in the terminal. Run `cd sign_language_knn` if needed.

---

## How it works

```
Webcam → MediaPipe (21 hand landmarks × 3 coords = 63 features)
      → StandardScaler → KNN Classifier → Predicted Sign
```

MediaPipe detects your hand and extracts 21 landmarks (knuckles, fingertips, wrist).
Each frame gives a 63-dimensional feature vector (x, y, z per point).
The KNN classifier finds the K nearest training samples and votes on the sign —
directly applying the Nearest Neighbor rule from the MML syllabus.

---

## Syllabus coverage (MML Non-Parametric Techniques)

| Concept | Where used |
|---|---|
| K-Nearest Neighbor rule | Core classifier |
| Euclidean distance metric | `metric="euclidean"` in KNN |
| K selection (elbow method) | `train_model.py` plots accuracy vs K |
| Feature extraction | MediaPipe landmarks as feature vector |
| Density estimation / smoothing | Majority vote over last 10 frames |
