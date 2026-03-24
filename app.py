from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import pickle
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
from tensorflow.keras.models import load_model
from gemini_helper import get_treatment_recommendations
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from twilio_helper import send_sms

app = Flask(__name__)
app.secret_key = 'agrisense_secret_key_2025'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def init_db():
    conn = sqlite3.connect('agrisense.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            n_value REAL,
            p_value REAL,
            k_value REAL,
            temperature REAL,
            humidity REAL,
            ph_value REAL,
            rainfall REAL,
            crop_type INTEGER,
            crop_days INTEGER,
            soil_moisture REAL,
            soil_temperature REAL,
            temperature2 REAL,
            humidity2 REAL,
            image_path TEXT,
            recommended_crop TEXT,
            crop_confidence REAL,
            irrigation_status TEXT,
            irrigation_confidence REAL,
            disease_name TEXT,
            disease_confidence REAL,
            gemini_recommendation TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def load_models():
    with open('models/crop_recommendation_model.pkl', 'rb') as f:
        crop_model = pickle.load(f)
    
    with open('models/irrigation_model.pkl', 'rb') as f:
        irrigation_model = pickle.load(f)
    
    with open('models/crop_label_encoder.pkl', 'rb') as f:
        crop_encoder = pickle.load(f)
    
    with open('models/crop_scaler.pkl', 'rb') as f:
        crop_scaler = pickle.load(f)
    
    with open('models/irrigation_scaler.pkl', 'rb') as f:
        irrigation_scaler = pickle.load(f)
    
    disease_model = load_model('models/plant_disease_model.h5')
    
    with open('models/class_names.pkl', 'rb') as f:
        disease_classes = pickle.load(f)
    
    return crop_model, irrigation_model, crop_encoder, crop_scaler, irrigation_scaler, disease_model, disease_classes

crop_model, irrigation_model, crop_encoder, crop_scaler, irrigation_scaler, disease_model, disease_classes = load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']
        
        hashed_password = generate_password_hash(password)
        
        conn = sqlite3.connect('agrisense.db')
        cursor = conn.cursor()
        
        try:
            # Using phone_number as column name to match DB update
            cursor.execute('INSERT INTO users (name, email, phone_number, password) VALUES (?, ?, ?, ?)',
                         (name, email, mobile, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            flash('Email already exists!', 'error')
            return redirect(url_for('signup'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect('agrisense.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            session['user_email'] = user[2]
            # Fetch phone_number (column 6)
            session['user_mobile'] = user[6] if len(user) > 6 else None
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password!', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user_name=session['user_name'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    n = float(request.form['nitrogen'])
    p = float(request.form['phosphorus'])
    k = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    crop_type = int(request.form['crop_type'])
    crop_days = int(request.form['crop_days'])
    soil_moisture = float(request.form['soil_moisture'])
    soil_temperature = float(request.form['soil_temperature'])
    temperature2 = float(request.form['temperature2'])
    humidity2 = float(request.form['humidity2'])
    
    file = request.files['leaf_image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{session['user_id']}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        flash('Invalid image file!', 'error')
        return redirect(url_for('dashboard'))
    
    crop_input = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    crop_input_scaled = crop_scaler.transform(crop_input)
    crop_prediction = crop_model.predict(crop_input_scaled)
    crop_proba = crop_model.predict_proba(crop_input_scaled)[0]
    recommended_crop = crop_encoder.inverse_transform(crop_prediction)[0]
    crop_confidence = max(crop_proba) * 100
    
    irrigation_input = np.array([[crop_type, crop_days, soil_moisture, soil_temperature, temperature2, humidity2]])
    irrigation_input_scaled = irrigation_scaler.transform(irrigation_input)
    irrigation_prediction = irrigation_model.predict(irrigation_input_scaled)
    irrigation_proba = irrigation_model.predict_proba(irrigation_input_scaled)[0]
    irrigation_status = 'Irrigate Now' if irrigation_prediction[0] == 1 else 'No Irrigation Needed'
    irrigation_confidence = max(irrigation_proba) * 100
    
    img = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    disease_prediction = disease_model.predict(img_array, verbose=0)
    disease_idx = np.argmax(disease_prediction)
    disease_confidence = np.max(disease_prediction) * 100
    disease_name = disease_classes[disease_idx]
    
    gemini_recommendation = get_treatment_recommendations(disease_name, recommended_crop, irrigation_status)
    
    conn = sqlite3.connect('agrisense.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (
            user_id, n_value, p_value, k_value, temperature, humidity, ph_value, rainfall,
            crop_type, crop_days, soil_moisture, soil_temperature, temperature2, humidity2,
            image_path, recommended_crop, crop_confidence, irrigation_status, irrigation_confidence,
            disease_name, disease_confidence, gemini_recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session['user_id'], n, p, k, temperature, humidity, ph, rainfall,
        crop_type, crop_days, soil_moisture, soil_temperature, temperature2, humidity2,
        filepath, recommended_crop, crop_confidence, irrigation_status, irrigation_confidence,
        disease_name, disease_confidence, gemini_recommendation
    ))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    session['prediction_id'] = prediction_id
    
    # Send SMS Notification
    if 'user_mobile' in session and session['user_mobile']:
        sms_body = (
            f"AGRISENSE Report:\n"
            f"Crop: {recommended_crop}\n"
            f"Irrigation: {irrigation_status}\n"
            f"Disease: {disease_name}\n"
            f"Check dashboard for details."
        )
        send_sms(session['user_mobile'], sms_body)
    
    return redirect(url_for('result'))

@app.route('/result')
def result():
    if 'user_id' not in session or 'prediction_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('agrisense.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions WHERE id = ?', (session['prediction_id'],))
    prediction = cursor.fetchone()
    conn.close()
    
    if not prediction:
        return redirect(url_for('dashboard'))
    
    result_data = {
        'timestamp': prediction[2],
        'recommended_crop': prediction[17],
        'crop_confidence': round(prediction[18], 2),
        'irrigation_status': prediction[19],
        'irrigation_confidence': round(prediction[20], 2),
        'disease_name': prediction[21],
        'disease_confidence': round(prediction[22], 2),
        'gemini_recommendation': prediction[23],
        'image_path': prediction[16]
    }
    
    return render_template('result.html', result=result_data, user_name=session['user_name'])

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('agrisense.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, recommended_crop, irrigation_status, disease_name 
        FROM predictions WHERE user_id = ? ORDER BY timestamp DESC
    ''', (session['user_id'],))
    predictions = cursor.fetchall()
    conn.close()
    
    history_data = []
    for pred in predictions:
        history_data.append({
            'id': pred[0],
            'timestamp': pred[1],
            'crop': pred[2],
            'irrigation': pred[3],
            'disease': pred[4]
        })
    
    return render_template('history.html', history=history_data, user_name=session['user_name'])

@app.route('/download_pdf')
def download_pdf():
    if 'user_id' not in session or 'prediction_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('agrisense.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions WHERE id = ?', (session['prediction_id'],))
    prediction = cursor.fetchone()
    conn.close()
    
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    p.setFont("Helvetica-Bold", 20)
    p.drawString(1*inch, height - 1*inch, "AGRISENSE - Farm Health Report")
    
    p.setFont("Helvetica", 12)
    y = height - 1.5*inch
    
    p.drawString(1*inch, y, f"Date: {prediction[2]}")
    y -= 0.3*inch
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(1*inch, y, "Crop Recommendation")
    y -= 0.25*inch
    p.setFont("Helvetica", 11)
    p.drawString(1*inch, y, f"Recommended Crop: {prediction[17]}")
    y -= 0.2*inch
    p.drawString(1*inch, y, f"Confidence: {prediction[18]:.2f}%")
    y -= 0.4*inch
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(1*inch, y, "Irrigation Advisory")
    y -= 0.25*inch
    p.setFont("Helvetica", 11)
    p.drawString(1*inch, y, f"Status: {prediction[19]}")
    y -= 0.2*inch
    p.drawString(1*inch, y, f"Confidence: {prediction[20]:.2f}%")
    y -= 0.4*inch
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(1*inch, y, "Disease Detection")
    y -= 0.25*inch
    p.setFont("Helvetica", 11)
    p.drawString(1*inch, y, f"Disease: {prediction[21]}")
    y -= 0.2*inch
    p.drawString(1*inch, y, f"Confidence: {prediction[22]:.2f}%")
    y -= 0.4*inch
    
    p.setFont("Helvetica-Bold", 14)
    p.drawString(1*inch, y, "Treatment Recommendations")
    y -= 0.25*inch
    p.setFont("Helvetica", 10)
    
    recommendation_text = prediction[23]
    lines = recommendation_text.split('\n')
    for line in lines:
        if y < 1*inch:
            p.showPage()
            y = height - 1*inch
        p.drawString(1*inch, y, line[:80])
        y -= 0.2*inch
    
    p.save()
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name=f'AgriSense_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf', mimetype='application/pdf')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('agrisense.db')
    # Get historical data for the user
    query = "SELECT timestamp, temperature, humidity, soil_moisture FROM predictions WHERE user_id = ?"
    df = pd.read_sql_query(query, conn, params=(session['user_id'],))
    conn.close()
    
    if df.empty:
        return render_template('analytics.html', analytics_data=None, user_name=session['user_name'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate Statistical Aggregations
    stats = {
        'temp': {'min': round(df['temperature'].min(), 2), 'max': round(df['temperature'].max(), 2), 'mean': round(df['temperature'].mean(), 2)},
        'hum': {'min': round(df['humidity'].min(), 2), 'max': round(df['humidity'].max(), 2), 'mean': round(df['humidity'].mean(), 2)},
        'soil': {'min': round(df['soil_moisture'].min(), 2), 'max': round(df['soil_moisture'].max(), 2), 'mean': round(df['soil_moisture'].mean(), 2)}
    }
    
    # Calculate Moving Averages (3-entry and 7-entry)
    # Note: Using entries as a proxy for days since timestamps might not be daily
    df['temp_ma3'] = df['temperature'].rolling(window=3).mean()
    df['temp_ma7'] = df['temperature'].rolling(window=7).mean()
    
    df['hum_ma3'] = df['humidity'].rolling(window=3).mean()
    df['hum_ma7'] = df['humidity'].rolling(window=7).mean()
    
    df['soil_ma3'] = df['soil_moisture'].rolling(window=3).mean()
    df['soil_ma7'] = df['soil_moisture'].rolling(window=7).mean()
    
    # Prepare data for Chart.js
    chart_data = {
        'labels': df['timestamp'].dt.strftime('%b %d, %H:%M').tolist(),
        'temp': {
            'actual': df['temperature'].tolist(),
            'ma3': df['temp_ma3'].fillna(0).tolist(),
            'ma7': df['temp_ma7'].fillna(0).tolist()
        },
        'hum': {
            'actual': df['humidity'].tolist(),
            'ma3': df['hum_ma3'].fillna(0).tolist(),
            'ma7': df['hum_ma7'].fillna(0).tolist()
        },
        'soil': {
            'actual': df['soil_moisture'].tolist(),
            'ma3': df['soil_ma3'].fillna(0).tolist(),
            'ma7': df['soil_ma7'].fillna(0).tolist()
        }
    }
    
    return render_template('analytics.html', stats=stats, chart_data=chart_data, user_name=session['user_name'])

@app.route('/send_sms', methods=['POST'])
def send_sms_route():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_mobile = session.get('user_mobile')
    if not user_mobile:
        flash('No mobile number found for this user.', 'error')
        return redirect(url_for('dashboard'))

    # Get prediction result from request (passed as hidden fields or reconstructed)
    # For simplicity, we'll construct a generic message or use the last prediction from DB if needed.
    # But ideally, we should pass the message content.
    # Let's assume we fetch the latest prediction for the user.
    
    try:
        conn = sqlite3.connect('agrisense.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (session['user_id'],))
        prediction = cursor.fetchone()
        conn.close()
        
        if prediction:
            # prediction columns: id, user_id, timestamp, ... recommended_crop (18), crop_confidence (19), irrigation_status (20), disease_name (23), gemini_rec (25)
            # Adjust indices based on CREATE TABLE if needed. 
            # Based on inspection:
            # 0: id, 1: user_id, 2: timestamp... 
            # 17: recommended_crop, 18: crop_confidence, 19: irrigation_status ...
            
            # Let's execute query to get specific columns to be safe
            conn = sqlite3.connect('agrisense.db')
            cursor = conn.cursor()
            cursor.execute('SELECT recommended_crop, irrigation_status, disease_name FROM predictions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1', (session['user_id'],))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                crop, irrigation, disease = row
                message_body = f"AGRISENSE Report:\nCrop: {crop}\nIrrigation: {irrigation}\nDisease: {disease}\nConsult the app for details."
                
                sid = send_sms(user_mobile, message_body)
                if sid:
                    flash(f'SMS sent successfully! SID: {sid}', 'success')
                else:
                    flash('Failed to send SMS. Check server logs.', 'error')
            else:
                flash('No prediction history found.', 'error')
        else:
            flash('No prediction history found.', 'error')
            
    except Exception as e:
        print(f"SMS Error: {e}")
        flash('An error occurred while sending SMS.', 'error')
        
    return redirect(url_for('dashboard')) # Or redirect back to result page if referrer is available

if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    init_db()
    app.run(debug=True)