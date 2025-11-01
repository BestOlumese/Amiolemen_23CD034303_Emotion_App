# Emotion Detection Web App (Amiolemen_23CD034303)

## Setup (local)
1. Clone / place folder.
2. Create virtualenv:
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)

3. Install requirements:
   pip install -r requirements.txt

4. Prepare training data (if training):
   - Create `data/train/<label>/*.jpg` and `data/test/<label>/*.jpg`
   - Run `python model.py` to train and save model as emotion_model.h5

5. Start app:
   python app.py
   Open http://127.0.0.1:5000/

6. Use the web UI to upload an image or use webcam.

## Notes
- Ensure `emotion_model.h5` exists before starting app.
- Ensure `emotion_labels` list in app.py matches the training class order.
- For deployment, consider saving the DB externally or using an RDS.
