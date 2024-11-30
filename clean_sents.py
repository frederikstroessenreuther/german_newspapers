def extract_windows_from_text(text):
    # Compile keywords into a regex pattern
    keyword_pattern = r'\b(' + '|'.join(re.escape(kw) for kw in KEYWORDS) + r')\b'
    # Find all matches with their positions
    matches = list(re.finditer(keyword_pattern, text, re.IGNORECASE))
    # Store non-overlapping windows
    windows = []
    used_indexes = set()
    for match in matches:
        # Get the start position of the match
        start_pos = match.start()
        # Check if this position has already been used
        if any(abs(start_pos - used) < WINDOW_SIZE * 10 for used in used_indexes):
            continue
        # Split text into words
        words = text.split()
        # Find the index of the keyword in the word list
        keyword_index = next(
            i for i, word in enumerate(words) 
            if re.search(keyword_pattern, word, re.IGNORECASE)
        )
        # Calculate window boundaries
        start_index = max(0, keyword_index - WINDOW_SIZE)
        end_index = min(len(words), keyword_index + WINDOW_SIZE + 1)
        # Extract window
        window = ' '.join(words[start_index:end_index])
        windows.append(window)
        used_indexes.add(start_pos)
    return windows



    
def analyze_text_windows(windows):
    # Sentiment analysis results
    sentiment_results = []
    for window in windows:
        if not window.strip():
            sentiment_results.append({
                'window': '',
                'sentiment_scores': [],
                'mean_sentiment': np.nan,
                'median_sentiment': np.nan
            })
            continue
        # Process the window text
        doc = nlp(window.strip())
        # Get sentiment scores for the window
        window_sentiments = [
            token._.sentiws 
            for token in doc 
            if hasattr(token._, 'sentiws') and token._.sentiws is not None
        ]
        # Compile results
        sentiment_results.append({
            'window': window,
            'sentiment_scores': window_sentiments,
            'mean_sentiment': np.mean(window_sentiments) if window_sentiments else np.nan,
            'median_sentiment': np.median(window_sentiments) if window_sentiments else np.nan
        })
    return sentiment_results