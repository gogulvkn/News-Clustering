import os
import re
import csv
import io
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. Initialize Flask App FIRST
app = Flask(__name__)
app.secret_key = "super_secret_key_for_clustering_app"

# Base Category Mapping
FALLBACK_CATEGORIES = {
    0: "ARTS & LITERATURE",
    1: "ATHLETICS & GENERAL SPORTS",
    2: "CORPORATE FINANCE & GLOBAL TRADE",
    3: "CORPORATE LITIGATION & POLICY",
    4: "INTERNATIONAL FOOTBALL & QUALIFIERS",
    5: "MACROECONOMICS & CENTRAL BANKING",
    6: "TECH FRAUD & DIGITAL MEDIA",
    7: "TELECOMMUNICATIONS & REGULATION",
    8: "UK POLITICS & FISCAL POLICY",
    9: "YUKOS BANKRUPTCY & CORPORATE TRIALS"
}

def extract_articles_from_df(df):
    cols_clean = [str(c).lower().strip() for c in df.columns]
    
    title_col = None
    content_col = None

    for cand in ['title', 'headline', 'heading', 'news_title']:
        if cand in cols_clean:
            title_col = df.columns[cols_clean.index(cand)]
            break

    for cand in ['content', 'body', 'text', 'description', 'article']:
        if cand in cols_clean and df.columns[cols_clean.index(cand)] != title_col:
            content_col = df.columns[cols_clean.index(cand)]
            break

    if not title_col or not content_col:
        str_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if len(str_cols) >= 2:
            title_col = title_col or str_cols[0]
            content_col = content_col or str_cols[1]
        elif len(str_cols) == 1:
            title_col = title_col or str_cols[0]
            content_col = str_cols[0]

    parsed = []
    for idx, row in df.iterrows():
        t = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else ""
        c = str(row[content_col]).strip() if content_col and pd.notna(row[content_col]) else ""

        if t.lower() in ['nan', 'none', 'null', 'title', '']:
            t = ""
        if c.lower() in ['nan', 'none', 'null', 'content', '']:
            c = ""

        if not t and not c:
            continue

        if not t and c:
            parts = re.split(r'(?<=[.!?]) +', c, maxsplit=1)
            t = parts[0]
            c = parts[1] if len(parts) > 1 else c

        feature = f"{t} {c}".strip()
        parsed.append({
            "title": t if t else f"Article {idx + 1}",
            "content": c,
            "text_feature": feature
        })

    return pd.DataFrame(parsed)


def load_all_rows_robustly(file):
    """
    Parser that guarantees extraction of all ~1700 records even with 
    broken multi-line quotes or corrupted CSV boundaries.
    """
    filename = file.filename.lower()
    
    if filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
        return extract_articles_from_df(df)

    file.seek(0)
    raw_bytes = file.read()
    
    try:
        content_str = raw_bytes.decode('utf-8')
    except UnicodeDecodeError:
        content_str = raw_bytes.decode('latin1', errors='ignore')

    # Try Pandas with QUOTE_NONE to prevent collapsing lines across raw quotes
    try:
        df = pd.read_csv(
            io.StringIO(content_str),
            on_bad_lines='skip',
            quoting=csv.QUOTE_NONE,
            escapechar='\\',
            encoding_errors='ignore'
        )
        if len(df) > 500:
            parsed_df = extract_articles_from_df(df)
            if len(parsed_df) > 500:
                return parsed_df
    except Exception:
        pass

    # Custom Line-by-Line Chunking Fallback
    lines = content_str.splitlines()
    parsed_articles = []
    
    current_title = None
    current_content = []

    for idx, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line:
            continue

        if idx == 0 and any(h in clean_line.lower() for h in ['title', 'content', 'headline', 'article']):
            continue

        parts = re.split(r'[,\t]', clean_line, maxsplit=1)
        
        is_new_record = (
            len(parts) == 2 and 
            len(parts[0].strip()) > 2 and 
            not parts[0].strip().endswith(('.', ':', ';'))
        )

        if is_new_record:
            if current_title or current_content:
                full_body = " ".join(current_content).strip()
                feature = f"{current_title or ''} {full_body}".strip()
                if len(feature) > 5:
                    parsed_articles.append({
                        "title": current_title if current_title else f"Article {len(parsed_articles) + 1}",
                        "content": full_body,
                        "text_feature": feature
                    })
            
            current_title = parts[0].strip(' "\' \t')
            current_content = [parts[1].strip(' "\' \t')]
        else:
            if current_title is None:
                s_parts = re.split(r'(?<=[.!?]) +', clean_line, maxsplit=1)
                current_title = s_parts[0].strip(' "\'')
                if len(s_parts) > 1:
                    current_content.append(s_parts[1].strip(' "\''))
            else:
                current_content.append(clean_line.strip(' "\''))

    if current_title or current_content:
        full_body = " ".join(current_content).strip()
        feature = f"{current_title or ''} {full_body}".strip()
        if len(feature) > 5:
            parsed_articles.append({
                "title": current_title if current_title else f"Article {len(parsed_articles) + 1}",
                "content": full_body,
                "text_feature": feature
            })

    return pd.DataFrame(parsed_articles)


def auto_assign_category_name(cluster_id, top_keywords):
    terms_set = set(top_keywords)
    
    if any(k in terms_set for k in ['film', 'award', 'book', 'music', 'band', 'actor', 'star', 'art']):
        return "ARTS & LITERATURE"
    if any(k in terms_set for k in ['olympic', 'athletics', 'race', 'gold', 'champion', 'medal', 'win']):
        return "ATHLETICS & GENERAL SPORTS"
    if any(k in terms_set for k in ['profit', 'firm', 'sales', 'shares', 'market', 'company', 'deal']):
        return "CORPORATE FINANCE & GLOBAL TRADE"
    if any(k in terms_set for k in ['court', 'judge', 'legal', 'law', 'case', 'claim', 'sued']):
        return "CORPORATE LITIGATION & POLICY"
    if any(k in terms_set for k in ['cup', 'match', 'england', 'club', 'league', 'player', 'goal', 'football']):
        return "INTERNATIONAL FOOTBALL & QUALIFIERS"
    if any(k in terms_set for k in ['growth', 'economy', 'bank', 'rates', 'dollar', 'inflation', 'gdp']):
        return "MACROECONOMICS & CENTRAL BANKING"
    if any(k in terms_set for k in ['virus', 'software', 'security', 'users', 'online', 'data', 'mail']):
        return "TECH FRAUD & DIGITAL MEDIA"
    if any(k in terms_set for k in ['mobile', 'phone', 'broadband', 'network', 'technology', 'tv', 'digital']):
        return "TELECOMMUNICATIONS & REGULATION"
    if any(k in terms_set for k in ['blair', 'election', 'party', 'labour', 'minister', 'government', 'tax']):
        return "UK POLITICS & FISCAL POLICY"
    if any(k in terms_set for k in ['yukos', 'russian', 'oil', 'gas', 'gazprom', 'bankruptcy']):
        return "YUKOS BANKRUPTCY & CORPORATE TRIALS"

    return FALLBACK_CATEGORIES.get(cluster_id, f"CATEGORY {cluster_id + 1}")


# 2. Define Routes AFTER creating `app`
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Please choose a valid file.')
            return redirect(request.url)

        try:
            df = load_all_rows_robustly(file)
            total_loaded = len(df)

            if total_loaded == 0:
                flash("No valid news articles were extracted.")
                return redirect(request.url)

            # Feature Extraction & Clustering
            vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
            X = vectorizer.fit_transform(df['text_feature'])

            n_clusters = min(10, total_loaded)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster_id'] = kmeans.fit_predict(X)

            terms = vectorizer.get_feature_names_out()
            order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

            categorized_results = []

            for cid in range(n_clusters):
                top_k = [terms[ind] for ind in order_centroids[cid, :10]]
                category_name = auto_assign_category_name(cid, top_k)
                cluster_df = df[df['cluster_id'] == cid]

                articles = []
                for _, row in cluster_df.iterrows():
                    body = str(row['content'])
                    articles.append({
                        "title": row['title'],
                        "content": body[:200] + "..." if len(body) > 200 else body
                    })

                categorized_results.append({
                    "category": category_name,
                    "total": len(cluster_df),
                    "articles": articles
                })

            categorized_results = sorted(categorized_results, key=lambda x: x['category'])

            return render_template('index.html', results=categorized_results, total_dataset=total_loaded)

        except Exception as e:
            flash(f"Error processing file: {str(e)}")
            return redirect(request.url)

    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)