import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


cat_ex = ['comp.graphics', 'comp.windows.x', 'rec.sport.baseball', 'sci.med', 'talk.politics.guns']
news = datasets.fetch_20newsgroups(categories=cat_ex)
train = datasets.fetch_20newsgroups(subset='train', categories=cat_ex)
test = datasets.fetch_20newsgroups(subset='test', categories=cat_ex)


def text_prep(text_data, *, data='train', words_num=(1, 1), frequency=True, inv_frequency=True):
    '''Preprocess text data (extracting features from text)'''
    if data == 'train':
        CV = CountVectorizer(analyzer='word', ngram_range=words_num)                # init token counts vectorizer
        tf_idf = TfidfTransformer(smooth_idf=frequency, use_idf=inv_frequency)      # init term frequency transformer
        word_counts = CV.fit_transform(text_data)                                   # fit transform text data to feature vectors
        feats = tf_idf.fit_transform(word_counts)                                   # fit transform tf features
        text_prep.vectorizer = CV                                                   # save vectorizer to attribute
        text_prep.transformer = tf_idf                                              # save transformer to attribute
    else:
        word_counts = text_prep.vectorizer.transform(text_data)                     # get transform text data to feature vectors
        feats = text_prep.transformer.transform(word_counts)                        # get transform tf features
    return text_prep.vectorizer, feats


def text_analyse(CV, feats):
    '''Displays the results of the analyzer'''
    f_names = CV.get_feature_names()
    for i, f in enumerate(feats):
        cx = f.tocoo()
        print('Sentence', i + 1, 'has:')
        for feature, count in zip(cx.col, cx.data):
            print('\t', count, f_names[feature])
        print(f.todense().tolist())


def main():
	example_text = ['This is the first sentence.', 
                'And this is the second sentence.', 
                'Is this the first sentence?']
	print(text_analyse(*text_prep(example_text, words_num=(2, 2))))

	X_train = text_prep(train.data, data='train')[1]
	X_test = text_prep(test.data, data='test')[1]
	Y_train, Y_test = train.target, test.target

	svm_clf_sigmoid = svm.SVC(kernel='sigmoid')
	svm_clf_sigmoid.fit(X_train, Y_train)
	svm_pred_sigmoid = svm_clf_sigmoid.predict(X_test)

	svm_clf_tanh = svm.SVC(kernel=lambda x, y: np.tanh(np.dot(x, y.T)))
	svm_clf_tanh.fit(X_train, Y_train)
	svm_pred_tanh = svm_clf_tanh.predict(X_test)

	nb_clf = MultinomialNB()
	nb_clf.fit(X_train, Y_train)
	nb_pred = nb_clf.predict(X_test)

	print(metrics.classification_report(test.target, svm_pred_sigmoid, target_names=test.target_names))
	print(metrics.classification_report(test.target, svm_pred_tanh, target_names=test.target_names))
	print(metrics.classification_report(test.target, nb_pred, target_names=test.target_names))


if __name__ == '__main__':
	main()