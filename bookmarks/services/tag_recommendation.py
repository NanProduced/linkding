import re
import math
from collections import Counter

from django.contrib.auth.models import User

from bookmarks.models import Tag, Bookmark


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
        'this', 'that', 'these', 'those', 'it', 'its', 'if', 'then', 'else',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
        'here', 'there', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'under', 'again', 'further',
        'what', 'which', 'who', 'whom', 'your', 'you', 'i', 'me', 'my', 'we',
        'our', 'us', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'ten', 'first', 'second', 'third', 'last', 'next', 'previous', 'new',
        'old', 'good', 'bad', 'best', 'better', 'worse', 'worst', 'many',
        'much', 'more', 'most', 'less', 'least', 'use', 'using', 'used', 'get',
        'getting', 'got', 'make', 'making', 'made', 'take', 'taking', 'took',
        'see', 'seeing', 'saw', 'know', 'knowing', 'knew', 'think', 'thinking',
        'thought', 'look', 'looking', 'looked', 'want', 'wanting', 'wanted',
        'give', 'giving', 'gave', 'find', 'finding', 'found', 'tell', 'telling',
        'told', 'ask', 'asking', 'asked', 'seem', 'seeming', 'seemed', 'feel',
        'feeling', 'felt', 'try', 'trying', 'tried', 'leave', 'leaving', 'left',
        'call', 'calling', 'called', 'keep', 'keeping', 'kept', 'let', 'letting',
        'begin', 'beginning', 'began', 'show', 'showing', 'showed', 'hear',
        'hearing', 'heard', 'play', 'playing', 'played', 'run', 'running', 'ran',
        'move', 'moving', 'moved', 'like', 'likes', 'liked', 'come', 'coming',
        'came', 'help', 'helping', 'helped', 'go', 'going', 'went', 'read',
        'reading', 'write', 'writing', 'written', 'learn', 'learning', 'learned',
        'change', 'changing', 'changed', 'turn', 'turning', 'turned', 'start',
        'starting', 'started', 'work', 'working', 'worked', 'study', 'studying',
        'studied', 'need', 'needing', 'needed', 'become', 'becoming', 'became',
        'put', 'putting', 'set', 'setting', 'mean', 'meaning', 'meant', 'keep',
        'say', 'saying', 'said', 'way', 'thing', 'time', 'man', 'woman', 'world',
        'life', 'hand', 'part', 'child', 'eye', 'woman', 'man', 'place', 'case',
        'week', 'company', 'system', 'program', 'question', 'work', 'government',
        'number', 'night', 'point', 'home', 'water', 'room', 'mother', 'area',
        'money', 'story', 'fact', 'month', 'lot', 'right', 'study', 'book',
        'eye', 'job', 'word', 'business', 'issue', 'side', 'kind', 'four',
        'head', 'house', 'service', 'friend', 'father', 'power', 'hour', 'game',
        'line', 'end', 'member', 'law', 'car', 'city', 'community', 'name',
        'team', 'minute', 'idea', 'kid', 'body', 'information', 'back', 'parent',
        'face', 'others', 'level', 'office', 'door', 'health', 'person', 'art',
        'war', 'history', 'party', 'result', 'change', 'morning', 'reason',
        'research', 'girl', 'guy', 'moment', 'air', 'teacher', 'force', 'education',
        'foot', 'boy', 'age', 'policy', 'process', 'music', 'market', 'kind',
        'loss', 'value', 'interest', 'care', 'people', 'state', 'country'
    }
    return [token for token in tokens if token not in stop_words]


def compute_tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counter = Counter(tokens)
    total_count = len(tokens)
    return {word: count / total_count for word, count in counter.items()}


def build_corpus(user: User) -> list[dict]:
    bookmarks = Bookmark.objects.filter(owner=user).prefetch_related('tags')
    corpus = []
    for bookmark in bookmarks:
        text = f"{bookmark.title} {bookmark.description}"
        tokens = tokenize(text)
        tag_names = [tag.name.lower() for tag in bookmark.tags.all()]
        corpus.append({
            'tokens': tokens,
            'tags': tag_names
        })
    return corpus


def compute_df(corpus: list[dict]) -> dict[str, int]:
    df = Counter()
    for doc in corpus:
        unique_tokens = set(doc['tokens'])
        df.update(unique_tokens)
    return df


def compute_idf(df: dict[str, int], total_docs: int) -> dict[str, float]:
    idf = {}
    for word, doc_count in df.items():
        idf[word] = math.log((total_docs + 1) / (doc_count + 1)) + 1
    return idf


def compute_tfidf(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    tfidf = {}
    for word, tf_value in tf.items():
        tfidf[word] = tf_value * idf.get(word, 1.0)
    return tfidf


def get_existing_tags(user: User) -> list[str]:
    tags = Tag.objects.filter(owner=user).values_list('name', flat=True)
    return [tag.lower() for tag in tags]


def map_tokens_to_tags(tokens: list[str], existing_tags: list[str]) -> dict[str, list[str]]:
    token_to_tags = {}
    for token in tokens:
        matching_tags = []
        for tag in existing_tags:
            if token == tag or tag.startswith(token) or token.startswith(tag):
                matching_tags.append(tag)
        if matching_tags:
            token_to_tags[token] = matching_tags
    return token_to_tags


def recommend_tags(
    title: str,
    description: str,
    user: User,
    count: int = 5
) -> list[str]:
    existing_tags = get_existing_tags(user)
    
    if not existing_tags:
        return []
    
    text = f"{title or ''} {description or ''}"
    tokens = tokenize(text)
    
    if not tokens:
        return []
    
    token_to_tags = map_tokens_to_tags(tokens, existing_tags)
    
    if not token_to_tags:
        return []
    
    corpus = build_corpus(user)
    
    if not corpus:
        matched_tokens = list(token_to_tags.keys())
        tag_scores = Counter()
        for token in matched_tokens:
            for tag in token_to_tags[token]:
                tag_scores[tag] += 1
        return [tag for tag, _ in tag_scores.most_common(count)]
    
    df = compute_df(corpus)
    total_docs = len(corpus)
    idf = compute_idf(df, total_docs)
    
    tf = compute_tf(tokens)
    tfidf = compute_tfidf(tf, idf)
    
    tag_scores = Counter()
    for token, score in tfidf.items():
        if token in token_to_tags:
            for tag in token_to_tags[token]:
                tag_scores[tag] += score
    
    for doc in corpus:
        for tag in doc['tags']:
            if tag in tag_scores:
                tag_scores[tag] += 0.1
    
    recommended = [tag for tag, _ in tag_scores.most_common(count)]
    
    return recommended
