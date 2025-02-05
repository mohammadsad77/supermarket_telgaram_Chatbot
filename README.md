## Usage

Obtain Your Telegram Bot Token
Create a Bot:
Start a conversation with BotFather on Telegram.

Follow BotFather's Instructions:
Use the /newbot command to create a new bot.
BotFather will then ask you for a name and username for your bot.

Receive the Token:
Once your bot is created, BotFather will send you a message that contains your bot token. It will look similar to this:
"123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"


Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chat.py
```

If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```