def load_sentiment_analysis_dataset(data_path, seed=123):
    data_path = os.path.join(data_path, 'data_twitter')

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['R', 'D']:
        train_path = os.path.join(data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname)) as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'D' else 1)

    # Load the validation data.
    test_texts = []
    test_labels = []
    for category in ['R', 'D']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'D' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))
