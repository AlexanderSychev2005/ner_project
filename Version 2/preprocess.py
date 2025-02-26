import spacy
from spacy.tokens import DocBin, Span
import pickle

nlp = spacy.blank("en")

# Load Data
training_data = pickle.load(open('./data/TrainData.pickle', 'rb'))
testing_data = pickle.load(open('./data/TestData.pickle', 'rb'))


def create_docbin(data, filename):
    db = DocBin()

    for text, annotations in data:
        doc = nlp(text)
        ents = []

        for start, end, label in annotations.get('entities', []):
            if not isinstance(label, str) or label is None:
                print(f"⚠️ Skipping entity with invalid label: {label} in text: {text[start:end]}")
                continue

            if not (0 <= start < end <= len(text)):
                print(f"❌ Skipping entity with invalid span ({start}, {end}) in text: {text}")
                continue

                # 🔍 Вывод отладочной информации
            print(f"🔍 Trying char_span: '{text[start:end]}' | Start: {start}, End: {end}, Label: {label}")

            span = doc.char_span(start, end, label=label, alignment_mode="expand")

            if span is None:
                print(f"⚠️ char_span failed for '{text[start:end]}'. Checking token boundaries.")

                start_token = len([t for t in doc if t.idx <= start])
                end_token = len([t for t in doc if t.idx < end])

                print(f"Tokens: start_token={start_token}, end_token={end_token}")

                if start_token < end_token:
                    span = Span(doc, start_token, end_token, label=label)
                else:
                    print(f"❌ Skipped entity '{text[start:end]}' due to token misalignment.")
                    continue

            ents.append(span)

        doc.ents = ents
        db.add(doc)

    db.to_disk(filename)


# Создание обучающего и тестового набора
create_docbin(training_data, "./data/train.spacy")
create_docbin(testing_data, "./data/test.spacy")
