from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import datetime

# Set up the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from a JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the trained model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Instantiate the neural network and load the weights
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

# Function to save chat logs to a text file
def save_chat(user_message, bot_response):
    now = datetime.datetime.now()
    filename = "chat_log.txt"  # Using a fixed file name for logging
    with open(filename, 'a') as file:
        file.write(f"Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"User: {user_message}\n")
        file.write(f"Bot: {bot_response}\n")
        file.write("-" * 50 + "\n")

# Function to predict the intent of a given sentence using the neural network model
def predict_class(sentence):
    sentence = tokenize(sentence)             # Tokenize the input sentence
    X = bag_of_words(sentence, all_words)       # Convert tokens to a Bag of Words feature vector
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)                           # Forward pass through the network
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    return tag

# Function to select an appropriate response based on the predicted intent
def get_response(tag):
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand."

# Telegram command handler for the /start command
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hello! I am the supermarket chatbot. How can I assist you today?")

# Telegram message handler that processes each incoming message
async def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text
    tag = predict_class(user_message)   # Determine the message intent
    response = get_response(tag)          # Get a response based on the intent

    # Log the chat
    save_chat(user_message, response)

    # Reply to the user
    await update.message.reply_text(response)

# Your Telegram Bot Token (obtain from BotFather)
TELEGRAM_BOT_TOKEN = 'your_telegram_bot_token_here'


# Create the Application instance using the builder (for python-telegram-bot version 20+)
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

# Add command and message handlers to the application
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("🤖 Telegram bot is running...")
app.run_polling()
