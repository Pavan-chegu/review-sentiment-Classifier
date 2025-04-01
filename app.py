from flask import Flask, render_template, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

app = Flask(__name__)

# Constants
MAX_LEN = 200






# Load the model and tokenizer
try:
    model = load_model('sentiment_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print("Please ensure you have run the notebook and saved both sentiment_model.h5 and tokenizer.pickle files")

def clean_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep negative contractions
    text = re.sub(r'[^a-zA-Z\s\']+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle negative contractions
    text = text.replace('\'t', ' not')
    
    # Remove stopwords except 'not' and 'no'
    stop_words = set(stopwords.words('english')) - {'no', 'not'}
    words = [w for w in text.split() if w not in stop_words]
    
    return ' '.join(words)
# Define positive and negative words
POSITIVE_WORDS = {
    "amazing", "brilliant", "fantastic", "excellent", "masterpiece", "incredible", "outstanding", "engaging",
    "captivating", "enthralling", "emotional", "heartfelt", "touching", "inspiring", "uplifting", "thought-provoking",
    "intelligent", "well-executed", "well-directed", "well-acted", "powerful", "groundbreaking", "iconic",
    "unforgettable", "superb", "remarkable", "beautiful", "stunning", "visually impressive", "cinematic excellence",
    "breathtaking", "immersive", "riveting", "must-watch", "compelling", "well-paced", "exceptional", "genius",
    "flawless", "seamless", "masterfully crafted", "Oscar-worthy", "top-notch", "well-written", "perfectly cast",
    "charming", "delightful", "hilarious", "witty", "clever", "fun", "entertaining", "joyful", "heartwarming",
    "satisfying", "rewarding", "mesmerizing", "phenomenal", "mind-blowing", "next-level", "out-of-this-world",
    "legendary", "awe-inspiring", "spectacular", "unparalleled", "artistic", "deep", "poignant", "memorable",
    "electrifying", "soul-stirring", "rich storytelling", "layered", "nuanced", "well-balanced", "thoughtful",
    "grounded", "relatable", "authentic", "high-quality", "brilliantly executed", "visually stunning",
    "cinematic masterpiece", "edge-of-your-seat", "game-changing", "unmatched", "daring", "effortlessly engaging",
    "enchanting", "bold", "charismatic", "gripping", "thrilling", "timeless", "believable", "genuine",
    "well-crafted", "deeply moving", "innovative", "sensational", "highly recommended", "strong performances",
    "great chemistry", "compelling story", "visually breathtaking", "stunning cinematography", "perfectly paced",
    "incredible acting", "masterful storytelling", "engrossing", "edge-of-seat", "intensely satisfying","love"
}

NEGATIVE_WORDS = {
    "boring", "slow", "predictable", "cliché", "overrated", "disappointing", "unoriginal", "dull", "cringe",
    "forced", "confusing", "awful", "messy", "weak", "flat", "unrealistic", "cheesy", "annoying", "lame",
    "forgettable", "terrible", "unwatchable", "lifeless", "horrible", "flawed", "pointless", "stupid",
    "overlong", "bad acting", "poor writing", "waste of time", "incoherent", "frustrating", "lazy",
    "overdramatic", "underwhelming", "unfunny", "mediocre", "repetitive", "bland", "wooden", "shallow",
    "chaotic", "generic", "nonsensical", "too predictable", "low-quality", "overhyped", "dragging",
    "illogical", "forced humor", "annoying characters", "unconvincing", "overacted", "emotionless",
    "forgettable soundtrack", "poor execution", "unrealistic dialogue", "rushed ending", "too long",
    "no depth", "unnecessary twists", "lackluster", "too dark", "over-explained", "miscast", "overcomplicated",
    "too violent", "weak direction", "unengaging", "hate", "horrendous", "atrocious", "pathetic", "disgusting",
    "detestable", "abominable", "repulsive", "appalling", "offensive", "insulting", "cringeworthy", "intolerable",
    "unbearable", "nauseating", "painful", "excruciating", "mind-numbing", "sickening", "grating", "abhorrent",
    "ridiculous", "dreadful", "infuriating", "meaningless", "soulless", "depressing", "tiresome",
    "tedious", "frustratingly bad", "garbage", "dumpster fire", "zero effort", "wretched", "irritating", "shoddy",
    "atrocity", "nightmarish", "shameful", "laughable", "crappy", "pathetically bad", "eye-roll inducing", "painstakingly bad",
    "poorly executed", "clumsy", "tone-deaf", "pointlessly edgy", "unnecessarily dramatic", "overblown", "exaggerated",
    "try-hard", "awkward", "corny", "ham-fisted", "unintentionally funny", "farcical", "botched", "silly in a bad way",
    "dragging pace", "overacted performances", "misused potential", "horrible CGI", "poor character development",
    "flat dialogue", "overstuffed plot", "pointlessly stretched", "failed execution", "script issues",
    "annoying soundtrack", "unconvincing performances", "choppy editing", "lacking tension", "directionless",
}

def predict_sentiment(text):
    """Predict sentiment using the trained model with sentiment word adjustment"""
    # Clean text
    cleaned_text = clean_text(text)
    words = set(cleaned_text.split())

    # Check sentiment words
    neg_count = len(words.intersection(NEGATIVE_WORDS))
    pos_count = len(words.intersection(POSITIVE_WORDS))

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

    # Predict
    prediction = float(model.predict(padded, verbose=0)[0][0])

    # Adjust prediction based on sentiment words
    if neg_count > pos_count:
        prediction = prediction * 0.5
    elif pos_count > neg_count:
        prediction = min(prediction * 1.2, 1.0)

    # Determine sentiment and confidence
    if prediction > 0.8:
        sentiment = "Positive"
        confidence = prediction
    elif prediction < 0.2:
        sentiment = "Negative"
        confidence = 1 - prediction
    else:
        if prediction < 0.4:
            sentiment = "Slightly Negative"
        elif prediction > 0.6:
            sentiment = "Slightly Positive"
        else:
            sentiment = "Neutral"
        confidence = abs(prediction - 0.5) * 2

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "confidence_percentage": f"{confidence:.2%}",
        "raw_prediction": prediction
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint for sentiment analysis"""
    text = request.form.get('text', '')
    
    if not text:
        return jsonify({"error": "Please enter some text"})
    
    try:
        result = predict_sentiment(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True) 