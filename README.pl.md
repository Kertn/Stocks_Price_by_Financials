
# Stock Price By Financials  
Budowanie modeli uczenia maszynowego do prognozowania ceny akcji na podstawie sprawozdań finansowych spółek

## Wprowadzenie

### Cel i główna idea projektu
Na rynku papierów wartościowych często obserwuje się akcje, których cena w krótkim czasie ulega znacznym zmianom. Zwykle wiąże się to z obiektywnymi, istotnymi zmianami w działalności przedsiębiorstwa, jednak zdarzają się również sytuacje, gdy akcje zyskują nadmierną lub zbyt małą popularność. Akcje takie — jeśli zostaną odpowiednio wcześnie zauważone — mogą okazać się świetną inwestycją, gdy inni inwestorzy dostrzegą ich nieuczciwą wycenę. Celem projektu jest odnajdywanie właśnie takich (niedowartościowanych lub przewartościowanych) akcji za pomocą modeli uczenia maszynowego.

### Testowanie modeli
Selekcję najlepszych modeli przeprowadzamy poprzez „pseudo-inwestycje” w akcje (na okres 1 roku), których cena według modelu jest nieadekwatna. Następnie program uwzględnia zmianę ceny akcji w kolejnym roku i wylicza roczną stopę zwrotu uzyskaną przez dany model.

#### Subtelności implementacyjne
1. Idea trenowania polega na porównaniu wyników finansowych spółki z jej ceną akcji. Model potrafi więc zidentyfikować akcje, których cena wyraźnie odbiega od innych o podobnej kondycji finansowej, lecz nie ocenia stopnia „przegrzania” całego rynku.  
2. Serwis finance.yahoo udostępnia jedynie 4 ostatnie sprawozdania finansowe, dlatego analizujemy inwestycje krótkoterminowe (1 rok).  
3. Dane są aktualne wyłącznie dla 2025 r.; w przyszłości należy je odświeżać i ponownie wyłonić najlepsze modele.

## Realizacja projektu

### Zbieranie danych
1. Ceny akcji i raporty finansowe pobieramy z finance.yahoo przy użyciu biblioteki YFinance. Listę wszystkich aktywnych spółek wraz z sektorami (All_lists) pozyskaliśmy, parsując stronę „macrotrends” (skrypt `data_collect.py`).  
2. Na podstawie tych list, znów z użyciem YFinance, utworzyliśmy bazy sprawozdań finansowych (All_lists_collected). W trakcie zbierania napisano też funkcje filtrujące dane nieistotne (`main.py`).

### Trenowanie modeli
Wybraliśmy siedem algorytmów ML: **BayesianRidge**, **ElasticNet**, **Random Forest**, **XgBoosting**, **Gradient Boosting**, **CatBoosting**, **Sieć Neuronowa**. Do optymalizacji hiperparametrów wykorzystano Optunę (`train_models.py`). Skuteczność oceniamy poprzez procentową różnicę między przewidywaną a rzeczywistą ceną akcji.

### Przygotowanie danych
• Usuwamy spółki/raporty z wieloma brakami danych.  
• Pozostałe luki wypełniamy „medianą z grupy o podobnej cenie akcji”.  
• Aby zmniejszyć liczbę zmiennych (~200), wybieramy te najmocniej skorelowane z ceną (hiperparametr `nlarge`).  
• Po tych krokach trenujemy modele z najlepszymi parametrami.

### Pseudo-inwestowanie
Model przewiduje ceny wszystkich spółek w danym sektorze. Jeżeli bieżąca cena jest niższa od prognozowanej o co najmniej `price_discount` %, skrypt „kupuje” `n` akcji. Po roku porównujemy ceny i obliczamy zysk/stratę, biorąc pod uwagę kierunek inwestycji (long/short) (`estimate_func.py`).  
Najlepsze hiperparametry i modele zapisywane są do pliku, a skrypt uruchamia się ponownie na danych z przedostatniego roku dla weryfikacji.

## Wyniki

### Tabela wyników
Pełną tabelę zawiera plik **Models_Results.xlsx**. Poniżej siedem czołowych modeli.  
![Top7](https://github.com/Kertn/Stock_Price_by_Financials/assets/111581848/966c5ecc-c2d1-4fa1-b20d-e7273af68278)

1. Sektor inwestycji  
2. Model  
3. Roczny dochód modelu  
4. Liczba kupionych akcji różnych spółek  
5. Łączna liczba dostępnych akcji różnych spółek  
6. Roczny dochód modelu (rok przedostatni)  
7. Liczba kupionych akcji (rok przedostatni)  
8. Liczba dostępnych akcji (rok przedostatni)  
9. Różnica % między ceną prognozowaną a rzeczywistą (100 % − min)  
10. Kierunek inwestycji  
11. Wymagany dyskont zakupu  
12. Parametry modelu  

### Inwestowanie
Dla 2025 r. najlepszy wynik osiągnął model (Medical Multi-Sector Conglomerates, CatBoost) z stopą zwrotu 36,8 % (19 kwietnia 2025).  
Kolejny to (Full_list, Random_Forest) z 1 %.  
Najgorzej: (Transportation, Sieć Neuronowa) −26,2 %.  
Łącznie 5 z 7 modeli przyniosło stratę od −11 % do −26 %.

### Wnioski
Ze względu na liczbę nieudanych inwestycji od początku 2025 r. warto podchodzić sceptycznie do automatycznego kupowania akcji według tych modeli — przynajmniej do poznania wyników za bieżący rok lub, lepiej, za 2026. Niemniej przewidywania (zwłaszcza CatBoost) dobrze wspierają ręczną analizę najbardziej niedowartościowanych — zdaniem modelu — spółek.
