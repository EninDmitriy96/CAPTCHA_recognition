
# CAPTCHA Recognition Project

  

This project implements two approaches for recognizing CAPTCHA images using a **pretrained ResNet-50** architecture. A **Streamlit frontend** and a **FastAPI backend** are provided for easy interaction with the models.

  

---

  

## Features

  

1.  **Two Recognition Approaches**:

-  **One-go Recognition**: Recognizes the entire CAPTCHA image in one step.

-  **Sequential Recognition**: Divides the CAPTCHA into individual characters and uses five classifiers to recognize each character sequentially.

  

2.  **Frontend and Backend**:

-  **Streamlit Frontend**: Provides an interface to upload CAPTCHA images and view recognition results.

-  **FastAPI Backend**: Handles predictions and model interaction.

  

3.  **Custom Dataset Training**:

- Users can train the models on their own datasets by configuring paths, CAPTCHA length, and alphabet.

  

4.  **Preprocessing Tools**:

- A script (`preprocess.py`) to clean the dataset by removing corrupted files.

  

---

  

## Installation and Setup

  

1. Clone the repository:

```bash
git clone https://github.com/EninDmitriy96/CAPTCHA_recognition
cd <path_to_the_project>
```

2. Install required dependencies:

  

```bash
pip  install  -r  requirements.txt
```

  

3. Ensure your dataset is prepared in the proper format:

CAPTCHA filenames should correspond to their decryption (e.g., abc12.png for CAPTCHA abc12).

Place CAPTCHA images in the data/ directory or configure paths for your dataset.

  

## Usage

  

### Running the Application

  

1. Start both the frontend and backend:

  

```bash
python  run_all.py
```

  

2.  Access  the  applications:

-  Frontend: [http://127.0.0.1:8501](http://127.0.0.1:8501)

-  Backend: [http://127.0.0.1:8000](http://127.0.0.1:8000)

  

3.  Terminate  both  processes  by  pressing  **Enter**  in  the  terminal.

  

---

  

### Using the Frontend

  

1.  **Upload  a  CAPTCHA  image**:

-  Use  examples  from  the  `data/`  folder  for  best  results (same dataset  used  for  training).

  

2.  **View  Predictions**:

-  The  results  from  both  approaches (one-go and  sequential  recognition) are displayed.

  

---

  

### Training on Your Dataset

  

1.  Update  the  following  parameters  in  the  code:

-  **Dataset  Paths**:  Ensure  paths  to  your  dataset  are  correct.

-  **Sequential  Classifiers**:  Set  the  number  of  classifiers  according  to  the  CAPTCHA  length.

-  **Alphabet**:  Define  the  alphabet  used  in  your  CAPTCHAs.

  

2.  Run  the  training  script (`code/models/notebooks`):

  

---

  

### Preprocessing the Dataset

  

To  remove  corrupted  files  from  your  dataset,  run:

  

```bash
python code/datasets/preprocess.py
```

  

## Future Improvements

  

-  Add  support  for  more  CAPTCHA  formats.

-  Improve  accuracy  for  custom  datasets.

-  Extend  frontend  for  additional  functionalities (e.g., batch  processing).

  

## Contributing

  

Contributions  are  welcome!  Please  open  an  issue  or  submit  a  pull  request  for  any  improvements  or  bug  fixes.

  

---
