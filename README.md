# Drzewa Decyzyjne
## Opis projektu
Projekt zawiera implementację klasyfikatora drzewa decyzyjnego przy użyciu biblioteki scikit-learn. Klasyfikatory są trenowane i oceniane na dwóch zestawach danych: zestawie danych Iris oraz zestawie danych Breast Cancer.
## Wnioski
Obserwując wysokości drzew na zestawie danych Iris możemy dojść do wniosku, że wskaźnik Information Gain lepiej sklasyfikował obiekty testowe (~92%), mimo że osiągnał większą wysokość drzewa.
Obserwując wysokości drzew na zestawie danych Breast Cancer, dochodzimy do wniosku że tym razem lepiej poradził sobie wskaźnik Gini, poprawnie sklasyfikował 94% obiektów testowych, mimo że jego drzewo było bardziej skomplikowane i miało wysokość równą 8. Dla tego zestawu danych wysokość drzewa Information Gain wyniosła 7, ale spadła dokładność klasyfikacji.

Na podstawie tych przykładów możemy zaobserwować, że często wraz z wzrostem wysokości drzewa rosnie dokładność klasyfikacji obiektów testowych.
