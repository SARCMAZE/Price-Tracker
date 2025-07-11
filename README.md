🛒 Myntra Price Tracker & Predictor 📉🔮
This project is a Myntra Price Tracker and Price Predictor that combines web scraping, WhatsApp notifications, and a price prediction model using machine learning. It is built to help users monitor price drops of their favorite products on Myntra and get predictions of future prices.

🚀 Features
✅ Scrapes product prices from Myntra using custom logic
🔔 Sends real-time WhatsApp alerts when price drops
📈 Predicts future prices using ARIMA or ML model
🔐 Environment variables managed securely using .env
🧪 Includes test files for environment and WhatsApp integration
🧾 Project Structure
myntra-tracker/ ├── .env # Environment variables (Twilio, URLs, etc.) ├── main.py # Main execution script ├── model.py # ML model for price prediction ├── myntra_price_predictor.py # ARIMA-based price prediction logic ├── scraper.py # Scrapes product data from Myntra ├── whatsapp.py # Sends WhatsApp messages via Twilio ├── test_env.py # Tests if .env variables are loading correctly ├── test_whatsapp.py # Tests WhatsApp message sending

yaml Copy Edit

⚙️ Technologies Used
Python 3.8+
BeautifulSoup & Requests – Web scraping
Twilio API – WhatsApp messaging
Pandas, NumPy – Data manipulation
Statsmodels / Scikit-learn – Price prediction
Dotenv – Secure environment variable handling
🛠️ Setup Instructions
Clone the repository
git clone https://github.com/your-username/myntra-tracker.git
cd myntra-tracker
Install required packages

bash
Copy
Edit
pip install -r requirements.txt
Configure .env file
Create a .env file in the root folder and add your credentials:

env
Copy
Edit
TWILIO_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE=your_twilio_number
TARGET_PHONE=your_whatsapp_number
PRODUCT_URL=https://www.myntra.com/product-link
Run the main script

bash
Copy
Edit
python main.py
🧪 Testing
To test environment variable setup:

bash
Copy
Edit
python test_env.py
To test WhatsApp notification:

bash
Copy
Edit
python test_whatsapp.py
