import pandas as pd
import re

# Importation des différents jeux de données
train_df = pd.read_csv("./tweets_train.csv", sep=",", header=None,
                       skipinitialspace=True, quotechar='"').values.tolist()
dev_df = pd.read_csv("./tweets_dev.csv", sep=",", header=None,
                     skipinitialspace=True, quotechar='"').values.tolist()
test_df = pd.read_csv("./tweets_test.csv", sep=",", header=None,
                      skipinitialspace=True, quotechar='"').values.tolist()


class Tweets_Processing():
    """
    Cette classe sert à traiter les tweets.

    Attributs :
        - tweets (list) : Liste des tweets.
        - nb_tweets (int) : Nombre de tweets.
        - default_stop_words (list) : Liste des stopwords par défaut (fichier importé).
        - smileys (list) : Liste des smileys.
        - mots (list) : Liste des mots différents dans les tweets.
        - n_mots (int) : Nombre de mots différents dans les tweets.
        - occurs (list) : Liste des occurrences des mots différents dans les tweets.

    Méthodes :
        - affichage(nb) : Affiche les 'nb' premiers tweets.
        - rm_punctuation() : Supprime la ponctuation et d'autres caractères indésirables des tweets.
        - different_words() : Retourne un dictionnaire des mots différents et de leurs occurrences dans les tweets.
        - rm_words() : Supprime les mots les plus fréquents et peu utilisés des tweets.
    """

    def __init__(self, tweets: list):
        """
        Initialisation de l'objet Tweets_Processing.

        Modifie chaque tweet de manière à ce qu'il soit traitable pour le calcul de probabilité.

        Arguments :
            - tweets (list) : Liste des tweets à traiter.
        """

        self.tweets = tweets  # Tweets à traiter

        self.nb_tweets = len(self.tweets)  # Le nombre de tweets initial

        self.default_stop_words = [
            word.rstrip('\n').lower() for word in open('english_stopwords.txt')
        ]  # Liste stopwords importés

        self.smileys = [
            ":)", ":(", ";)", ";(", ":d", ":D", ":p", ":P", ":o", ":O", ":@", ":/",
            ":\\", ":|", ":*", ":$", ":&", ":#", ":!", ":%", ":^", ":(", ":)", ";(",
            ";)", "<3", "</3", ":3", ";3", ":-)", ":-("
        ]  # Liste d'emojis

        # Appel de la méthode rm_punctuation pour supprimer la ponctuation des tweets
        self.rm_punctuation()

        # Liste des mots uniques présents dans les tweets
        self.mots = list(self.different_words().keys())
        # Nombre de mots uniques dans les tweets
        self.n_mots = sum([len(tweet.split(" ")) for tweet in self.tweets])
        # Liste des occurrences de chaque mot unique dans les tweets
        self.occurs = list(self.different_words().values())

    # Méthode que l'on a utilisé pour avoir des informations concernant les tweets
    def affichage(self, nb: int) -> None:
        """
        Affiche les nb premiers tweets.

        Arguments :
            - nb (int) : Nombre de tweets à afficher.
        """
        for tweet in range(nb):  # Pour chaque tweet
            print(self.tweets[tweet])  # On l'affiche

    def rm_punctuation(self) -> None:
        """
        Supprime la ponctuation des tweets tout en conservant les emojis.
        Effectue également d'autres étapes de prétraitement sur les tweets.
        """
        new_tweets = []

        for tweet in self.tweets:  # Pour chaque tweet

            tweet = tweet.lower()  # On convertit le texte du tweet en minuscules

            tweet = re.sub(r'http\S+', '', tweet)  # On supprime les URLs
            # On supprime les mentions (@utilisateur et @utilisateur:)
            tweet = re.sub(r'@\w+[:]*', '', tweet)

            words = []  # Liste pour stocker les mots du tweet

            for word in tweet.split():
                if word not in self.smileys:  # S'il ne s'agit pas d'un emoji
                    # Sépare les caractères spéciaux des mots (i've -> i ve)
                    word = re.sub(r'(\w)([^\w\s:])(\w)', r'\1 \2 \3', word)
                    # Supprime les caractères spéciaux restants (de' -> de)
                    word = re.sub(r'[^\w\s:]|\d', '', word)

                # On ajoute le mot à la liste des mots du tweet
                words.append(word)

            tweet = " ".join(words)  # On reforme le tweet
            # On supprime les caractères individuels et les chiffres
            tweet = " ".join([word for word in tweet.split() if len(word) > 1])
            # On supprime les mots à l'aide des stopwords importés
            tweet = " ".join([word for word in tweet.split()
                             if word not in self.default_stop_words])

            # On supprime les espaces multiples
            tweet = re.sub(r'\s+', ' ', tweet).strip()
            new_tweets.append(tweet)  # On ajoute le tweet edité à la liste

        self.tweets = new_tweets  # On remplace les tweets par les tweets traités

    def different_words(self) -> dict:
        """
        Calcule le nombre d'occurrences de chaque mot unique dans les tweets.

        Retourne :
            - Un dictionnaire (dict) contenant les mots uniques et leur nombre d'occurrences.
        """

        occ = {}  # On utilise un dictionnaire pour stocker les mots uniques et leurs occurrences

        for i in range(len(self.tweets)):  # Pour chaque tweet
            # On fait une liste des mots du tweet
            tweet_words = self.tweets[i].split(" ")

            for word in tweet_words:  # Pour chaque mot du tweet

                if word in occ:  # Si le mot existe déjà dans le dictionnaire, on incrémente son occurence
                    occ[word] += 1
                else:  # Si le mot n'existe pas dans le dictionnaire, on l'ajoute dans le dictionnaire
                    occ[word] = 1

        return occ

    def rm_words(self) -> None:
        """
        Supprime les 100 mots les plus fréquents des tweets.
        Les mots ayant une fréquence inférieure à 2 ou supérieure au seuil seront également supprimés.
        """

        # Dictionnaire contenant les mots uniques et leur nombre d'occurrences dans les tweets
        occ = self.different_words()

        # Liste contenant le nombre d'occurrences de chaque mot
        occus_nb = list(occ.values())
        # On trie la liste de maniere à ce que les mots avec le plus d'occurences soient au début
        occus_nb.sort(reverse=True)
        threshold = occus_nb[99]  # On définit la valeur du seuil

        for i in range(len(self.tweets)):  # Pour chaque tweet
            # On fait une liste des mots du tweet
            tweet_words = self.tweets[i].split(" ")

            for word in tweet_words:  # Pour chaque mot du tweet

                # Si le mot a une fréquence inférieure à 2 ou supérieure au seuil, il sera supprimé du tweet
                if occ[word] < 2 or occ[word] > threshold:
                    tweet_words.remove(word)

            self.tweets[i] = " ".join(tweet_words)  # On met à jour le tweet


####################################################################
# Partie TRAIN
####################################################################

# Initialisation des listes de tweets
pos = []
neg = []
corpus = []

# Ajout des tweets
for tweet in train_df:
    if tweet[0] == 'positive':
        pos.append(tweet[1])
    else:
        neg.append(tweet[1])
    corpus.append(tweet[1])

# Traitement des tweets à l'aide de la classe Tweets_Processing (pos, neg, corpus)
tweets_pos = Tweets_Processing(pos)
tweets_pos.rm_words()
tweets_neg = Tweets_Processing(neg)
tweets_neg.rm_words()
corpus = Tweets_Processing(corpus)
corpus.rm_words()

# Calcul des différentes probabilités (Tot, Proba pos, Proba neg)
p_pos = tweets_pos.nb_tweets
p_neg = tweets_neg.nb_tweets
tot = corpus.nb_tweets

p_pos = p_pos/tot  # Probabilité qu'un tweet soit pos sur corpus
p_neg = p_neg/tot  # Probabilité qu'un tweet soit neg sur corpus


# Nombre de mots
n_corp = corpus.n_mots  # nb de mots total
n_pos = tweets_pos.n_mots  # nb de mots pos sur corpus
n_neg = tweets_neg.n_mots  # nb de mots neg sur corpus

# Mots différents
mots_corp = corpus.mots  # liste de mots du corpus différents
mots_pos = tweets_pos.mots  # liste de mots pos différents
mots_neg = tweets_neg.mots  # liste de mots neg différents

# Occurences
occur_corp = corpus.occurs  # liste d'occurence des mots dans la totalité des tweets
occur_pos = tweets_pos.occurs  # liste d'occurence des mots pos des tweets
occur_neg = tweets_neg.occurs  # liste d'occurence des mots neg des tweets

####################################################################
# Utilisation des jeux de données
####################################################################


def use_data(dataset: list) -> tuple[list, list]:
    """
    Permet d'utiliser un fichier dataset pour pouvoir ensuite le traiter.

    Arguments :
        - dataset (list) : liste à 2 dimensions importée par pandas. 

    Retourne :
        - tweets (list) : liste des différents tweets.
        - labels (list) : liste des différents labels.
    """
    # On initialise les listes
    labels = []
    tweets = []

    for tweet in dataset:  # Pour chaque tweet du dataset
        labels.append(tweet[0])  # On ajoute le label à la liste
        tweets.append(tweet[1])  # On ajoute le tweet à la liste

    return tweets, labels


def prob_calc(tweet: str) -> str:
    """
    Permet de prédir le label d'un tweet (positive/negative).

    Arguments :
        - tweet (str) : tweet à utiliser.

    Retourne :
        - label (str) : retourne positive ou negative.
    """

    # On fait le traitement pour un unique tweet et on le convertit en liste de mots
    word_list = Tweets_Processing([tweet]).tweets[0].split(" ")
    # On initialise les différentes probabilités
    p_txt_pos = 1
    p_txt_neg = 1
    p_txt = 1

    for m in word_list:  # Pour chaque mot du tweet

        if m in mots_corp:  # S'il appartient aux mots du corpus
            p_m = occur_corp[mots_corp.index(m)]  # On récupère son occurence
            p_txt *= p_m  # On la multiplie avec p_txt

        if m in mots_pos:  # S'il appartient aux mots positifs
            p_m = occur_pos[mots_pos.index(m)]  # On récupère son occurence
            p_txt_pos *= p_m  # On la multiplie avec p_txt_pos

        if m in mots_neg:
            p_m = occur_neg[mots_neg.index(m)]  # On récupère son occurence
            p_txt_neg *= p_m  # On la multiplie avec p_txt_neg

    # Calcul des différentes probabilités
    p_txt /= n_corp
    p_txt_pos /= n_pos
    p_txt_neg /= n_neg
    p_pos_txt = (p_txt_pos*p_pos)/p_txt
    p_neg_txt = (p_txt_neg*p_neg)/p_txt

    if p_pos_txt > p_neg_txt:  # Si on voit que la probabilité que le tweet soit positif est plus grande
        return "positive"
    return "negative"  # Sinon on dit que le tweet est négatif


def accuracy(tweets: list, labels: list) -> float:
    """
    Permet de donner le pourcentage de prédiction correctes pour des tweets.

    Arguments :
        - tweets (list) : tweets à utiliser.
        - labels (list) : labels à utiliser.

    Retourne :
        - accuracy (float) : pourcentage de prédictions correctes.
    """
    right = 0  # On initialise le nombre de bonnes prédictions
    wrong = 0  # On initialise le nombre de mauvaises prédictions

    for t in range(len(tweets)):  # Pour chaque tweet
        # On vérifie si la probabilité calculée correspond
        if prob_calc(tweets[t]) == labels[t]:
            right += 1  # On incrémente le nombre de bonnes prédictions
        else:
            wrong += 1  # On incrémente le nombre de mauvaises prédictions
    return (right/(right+wrong))  # On calcule et retourne la prédiction


if __name__ == "__main__":
    print("###########################################################")
    print("Projet IS NLP")
    print("###########################################################\n")

    try:
        print("→ Calcul du pourcentage de prédictions correctes pour le jeu de données 'dev' :")
        dev = use_data(dev_df)
        print(f"Réussite : {round(accuracy(dev[0],dev[1]),3)*100} %\n")

        print("→ Calcul du pourcentage de prédictions correctes pour le jeu de données 'test' :")
        test = use_data(test_df)
        print(f"Réussite : {round(accuracy(test[0],test[1]),3)*100} %")
        print("\n###########################################################")
        print("Fin d'exécution du programme : Tout s'est bien passé :)")
        print("###########################################################\n")

    except Exception as e:
        print("\n###########################################################")
        print("Fin d'exécution du programme suite à une erreur : ")
        print(f"[ERREUR] : {e}")
        print("###########################################################\n")
