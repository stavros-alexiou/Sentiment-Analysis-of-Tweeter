# Sentiment-Analysis-of-Tweeter

Υπάρχουν πολλά σύνολα δεδοµένων ανάλυσης συναισθηµάτων (sentiment analysis) από το Twitter. Ένα από τα πιο δηµοφιλή σύνολα δεδοµένων ονοµάζεται 'sentiment140' , το οποίο περιέχει 1,6 εκατοµµύρια προεπεξεργασµένα Tweets. Αυτό είναι ένα εξαιρετικό σύνολο δεδοµένων για να ξεκινήσετε εάν είστε νέοι στην ανάλυση συναισθηµάτων. 

Αυτά τα Tweets έχουν σχολιαστεί και η µεταβλητή στόχος είναι το συναίσθηµα. Οι µοναδικές τιµές σε αυτήν τη στήλη είναι 0 (αρνητικό), 2 (ουδέτερο) και 4 (θετικό).

Τα tweets που προέκυψαν έχουν υποστεί κατάλληλη προεπεξεργασία και έχει γίνει η μετατροπή τους σε διανύσματα και χρησιμοποιήθηκαν για την τροφοδοσία των εισόδων του εκάστοτε νευρωνικού δικτύου ή αλγορίθμου μηχανικής μάθησης που υλοποιήθηκε.

Οι αλγόριθμοι που χρησιμοποιήθηκαν, στο παρόν repository, για την σύγκριση αποτελεσμάτων και τη διεξαγωγή συμπερασμάτων αναφέρονται ακολούθως:

Neural Networks:
1. Recurrent GRU.
2. Recurrent LSTM.
3. Multilayer Perceptron.

Machine Learning Algorithms:
1. Naive Bayes Classifier.
2. Decision Tree Classifier.
3. XGBoost Classifier.

Attribute Information:
1. target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive).
2. ids: The id of the tweet (2087).
3. date: the date of the tweet (Sat May 16 23:58:44 UTC 2009).
4. flag: The query (lyx). If there is no query, then this value is NO_QUERY.
5. user: the user that tweeted (robotickilldozr).
6. text: the text of the tweet (Lyx is cool).
