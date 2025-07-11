🛒 Myntra Price Tracker & Predictor 📉🔮
A smart price monitoring and prediction tool for Myntra, built with web scraping, machine learning, and real-time WhatsApp alerts. Track your favorite product’s price drops and predict future pricing trends with ease.

🚀 Features
✅ Live Price Scraper – Fetches product data directly from Myntra using a custom scraper.

🔔 WhatsApp Alerts – Sends real-time notifications for price drops via Twilio’s WhatsApp API.

📈 Future Price Prediction – Utilizes ARIMA or ML models (like Linear Regression) to forecast prices.

🔐 Secure Environment Handling – Keeps API credentials safe using .env files.

🧪 Test-Driven Setup – Includes testing scripts for environment configuration and WhatsApp integration.


⚙️ Technologies Used
Python 3.8+

BeautifulSoup & Requests – For scraping product data

Twilio API – For WhatsApp alerts

Pandas, NumPy – For data handling

Statsmodels / Scikit-learn – For time series and ML predictions

python-dotenv – For managing sensitive credentials securely

🛠️ Setup Instructions
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/myntra-tracker.git
cd myntra-tracker
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Configure Environment

Create a .env file in the project root:

env
Copy
Edit
TWILIO_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE=your_twilio_whatsapp_number
TARGET_PHONE=recipient_whatsapp_number
PRODUCT_URL=https://www.myntra.com/product-link
Run the Main Script

bash
Copy
Edit
python main.py
🧪 Testing
Test .env Configuration

bash
Copy
Edit
python test_env.py
Test WhatsApp Messaging

bash
Copy
Edit
python test_whatsapp.py
📌 Future Improvements
🧠 Deep learning-based price prediction (LSTM, Prophet, etc.)

📊 Dashboard UI to visualize price trends

🧵 Multi-product tracking with database support

📱 Telegram support (alternative to WhatsApp)

🤝 Contributing
Pull requests are welcome! If you’d like to add new features or fix bugs, feel free to fork the repo and submit a PR.

