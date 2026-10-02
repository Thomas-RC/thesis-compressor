# 5. Wyniki eksperymentów i analiza

Niniejszy rozdział stanowi podsumowanie przeprowadzonych prac badawczych oraz szczegółową analizę wydajności zaproponowanego kompresora neuronowego. Cel eksperymentów był dwojaki. Po pierwsze, dążono do empirycznego wykazania zdolności architektury Transformer operującej na poziomie surowych bajtów do estymowania rozkładu prawdopodobieństwa strumienia danych. Po drugie, zbadano granicę nasycenia pojemności informacyjnej modelu poprzez weryfikację praw skalowania w bezpośrednim starciu z referencyjnymi algorytmami słownikowymi oraz w obrębie własnej hierarchii konfiguracji.

## 5.1. Środowisko testowe i metodyka ewaluacji

Z uwagi na deterministyczny charakter obliczeń tensorowych oraz konieczność zapewnienia pełnej reprodukowalności wyników, ostateczna ewaluacja została przeprowadzona na dedykowanej stacji roboczej. Platformę testową stanowił akcelerator graficzny NVIDIA RTX 4080 Super dysponujący 16 gigabajtami pamięci VRAM. Wykorzystanie sprzętowej akceleracji poprzez interfejs CUDA pozwoliło na osiągnięcie przepustowości rzędu kilkunastu kroków optymalizatora na sekundę dla konfiguracji o rozmiarze 20 milionów parametrów.

| Komponent | Specyfikacja |
|---|---|
| Akcelerator graficzny | NVIDIA RTX 4080 Super, 16 GB VRAM |
| Procesor główny | wielordzeniowy x86-64 (operacje przygotowawcze) |
| Środowisko systemowe | Linux + WSL2 / native Linux |
| Stos technologiczny | Rust 1.83+, Candle 0.10, CUDA 12.x |
| Tryb precyzji | FP32 (pełna precyzja zmiennoprzecinkowa) |
| Korpus | enwik8 (90 MB train / 5 MB val / 5 MB test) |

*Tabela 5.1: Specyfikacja środowiska eksperymentalnego.*

Metodyka pomiarowa opierała się na rygorystycznym podziale korpusu enwik8 na trzy rozłączne podzbiory. Kluczowym aspektem ewaluacji końcowej było zastosowanie procedury deterministycznej na pełnym zbiorze testowym z wykorzystaniem rozłącznych okien kontekstowych (procedura non-overlapping). Pozwoliło to na uzyskanie stabilnych i wolnych od szumu statystycznego wartości metryki BPC, które stanowią jedyną podstawę do porównań z klasycznymi metodami kompresji. Dla każdego raportowanego punktu pomiarowego wartości train, validation oraz test BPC obliczane były na rozłącznych pulach danych, eliminując ryzyko wycieku informacji między fazami uczenia a ewaluacji końcowej.

## 5.2. Przeszukiwanie przestrzeni parametrów (Grid Search)

W celu precyzyjnego zmapowania wpływu architektury na dynamikę uczenia przeprowadzono badanie typu Grid Search obejmujące 27 niezależnych przebiegów treningowych w trybie jednolitego budżetu obliczeniowego (2000 kroków optymalizatora). Zmiennymi w badaniu były trzy hiperparametry definiujące skalę i horyzont modelu: wymiar osadzeń (`d_model` ∈ {128, 256, 512}), liczba warstw (`n_layers` ∈ {2, 4, 6}) oraz długość okna kontekstowego (`seq_len` ∈ {256, 512, 1024}).

### 5.2.1. Zjawisko niedotrenowania i wpływ horyzontu kontekstowego

Zestawienie przebiegów w reżimie stałego budżetu obliczeniowego ujawniło istotną hierarchię wydajności. Najlepsze rezultaty uzyskały architektury charakteryzujące się niewielką głębokością przy zachowaniu szerokich reprezentacji wektorowych. Konfiguracja optymalna (d=512, L=2, seq=1024) osiągnęła błąd walidacyjny na poziomie **4,05 BPC**.

*Rycina 5.1: Mapa ciepła obrazująca błąd walidacyjny BPC dla wszystkich 27 konfiguracji grid search w trzech panelach odpowiadających długości okna (256, 512, 1024 bajty). Widoczna degradacja wyniku dla modeli głębokich (L=6) przy ograniczonym budżecie treningowym.*

Analiza wykazała, że modele o większej liczbie warstw cierpią na zjawisko niedotrenowania w początkowej fazie optymalizacji — przy budżecie 2000 kroków głębsze sieci nie zdążają wykorzystać swojej pojemności reprezentacyjnej. Jednocześnie zaobserwowano spójną tendencję: wydłużenie okna kontekstowego z 256 do 1024 bajtów konsekwentnie poprawia wynik kompresji niezależnie od liczby parametrów. Zysk wynikający z poszerzenia horyzontu predykcyjnego wynosił od około 0,06 BPC dla najmniejszych konfiguracji do 0,08 BPC dla największych, co potwierdza krytyczne znaczenie odległych zależności w danych tekstowych typu Wikipedia.

*Rycina 5.2: Zestawienie wszystkich badanych konfiguracji w relacji do liczby parametrów. Czerwona przerywana linia wyznacza próg algorytmu GZIP (2,92 BPC). Wszystkie przebiegi grid search znajdują się w przedziale 4,05–4,56 BPC, sygnalizując niedostatecznie wykorzystany potencjał architektury w reżimie krótkiego treningu.*

## 5.3. Analiza długookresowych procesów uczenia i skalowania

Kluczowym etapem badań była weryfikacja praw skalowania dla większych konfiguracji sieciowych przy znacząco wydłużonym czasie ekspozycji na dane. W tej fazie eksperymentów uruchomiono cykle treningowe trwające do 40 000 kroków optymalizatora, co odpowiada kilkukrotnemu zwiększeniu budżetu obliczeniowego względem grid search.

### 5.3.1. Nasycenie pojemności architektury bazowej

Trening konfiguracji `large` (19,7 mln parametrów) na budżecie 656 milionów tokenów (40 000 kroków × batch 16 × seq_len 1024) obnażył zjawisko twardego nasycenia pojemności. Model osiągnął optymalny pułap w okolicach 12 000 kroku i pomimo dalszego uczenia oraz zastosowania fazy wygaszania współczynnika uczenia po kroku 32 000, nie zdołał przełamać bariery 4,03 BPC. Końcowa wartość zmierzona deterministycznie na zbiorze testowym wyniosła **4,034 BPC**.

*Rycina 5.3: Przebieg funkcji straty (train/val) w jednostkach BPC dla modelu `large`. Widoczne wyraźne plateau pomiędzy krokiem 5 000 a 32 000 oraz brak reakcji na liniową redukcję współczynnika uczenia w finalnej fazie wygaszania.*

### 5.3.2. Przewaga skalowania parametrów nad skalowaniem danych

Konfrontacja modelu `large` z wariantem `xlarge` (57,8 mln parametrów) dostarczyła dowodów na nieliniowy charakter zysków z kompresji w domenie byte-level. Model `xlarge` trenowany na **czterokrotnie mniejszym budżecie danych** (163 mln tokenów wobec 656 mln) osiągnął wynik 3,987 BPC, pokonując mniejszą architekturę z budżetem czterokrotnie większym. Potwierdza to tezę, że w reżimie modelowania bajtowego — przy zachowaniu rozsądnych proporcji — powiększanie struktury sieci stanowi czynnik istotniejszy niż dalsze zwiększanie wolumenu danych treningowych.

## 5.4. Ablacja architektoniczna: wpływ nowoczesnych komponentów

Ostatnia faza eksperymentów dotyczyła wpływu jakościowej zmiany topologii sieci na przełamanie zaobserwowanego architektonicznego plateau. Wdrożenie komponentów typowych dla architektury LLaMA (RMSNorm, RoPE, SwiGLU) pozwoliło na uzyskanie znaczącej poprawy wydajności bez zmiany budżetu obliczeniowego ani liczby parametrów.

Wariant `xlarge_llama` (57,0 mln parametrów, identyczna procedura treningowa: 20 000 kroków, batch 8) osiągnął na zbiorze testowym wynik **3,920 BPC**. Stanowi to bezwzględną redukcję błędu o 0,067 BPC w stosunku do wariantu `xlarge` z klasyczną architekturą GPT-2 (3,987 BPC). Udowadnia to, że ograniczenie na poziomie 4,0 BPC nie wynikało z fundamentalnych praw teorii informacji, lecz było artefaktem przestarzałej topologii oraz nieoptymalnego przepływu sygnału atencji w bazowym modelu. Szczegółowa analiza ablacyjna poszczególnych komponentów (RMSNorm, RoPE, SwiGLU oraz ich kombinacji) zaprezentowana została w odrębnym rozdziale 6.

## 5.5. Empiryczne prawa skalowania (Chinchilla Fit)

Na podstawie 29 punktów pomiarowych (27 konfiguracji grid search wzbogaconych o dwa długoterminowe przebiegi `large` i `xlarge`) wyznaczono autorskie parametry skalowania dla modelu bajtowego. Dopasowanie do funkcji potęgowej, znanej z prac Hoffmann i in. (2022), realizowano metodą najmniejszych kwadratów nieliniowych (`scipy.optimize.curve_fit`):

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

gdzie `L` oznacza stratę wyrażoną w natach na token (`BPC × ln 2`), `N` liczbę parametrów modelu, a `D` całkowitą liczbę tokenów przetworzonych podczas treningu (`steps × batch × seq_len`). Parametr `E` reprezentuje *irreducible loss* — teoretyczne minimum funkcji straty asymptotycznie osiągalne w nieskończonym reżimie skali.

Wyznaczone wartości parametrów dopasowania:

- **α (wykładnik dla N)**: 0,095
- **β (wykładnik dla D)**: 0,041
- **E (irreducible loss)**: 2,27 BPC

Wartość `E` poniżej poziomu 2,92 BPC (próg GZIP) stanowi matematyczny dowód na to, że badana architektura posiada teoretyczny potencjał do pokonania algorytmu GZIP przy zapewnieniu odpowiedniej skali parametrów oraz danych. Niska wartość wykładnika β (znacząco mniejsza od α) wyjaśnia jednocześnie zaobserwowaną wcześniej słabą reakcję modelu na zwiększanie samego budżetu tokenów — w domenie byte-level dominującą dźwignią optymalizacyjną pozostaje pojemność strukturalna sieci, a nie ilość danych. Należy zaznaczyć, że ze względu na ograniczoną liczbę punktów pomiarowych oraz zawężony zakres przebadanych skal (do około 6·10⁷ parametrów), wyznaczone współczynniki obarczone są istotną niepewnością i powinny być traktowane jako wskaźniki kierunkowe, a nie ekstrapolowalne prawo asymptotyczne.

## 5.6. Zestawienie wyników końcowych i granice ekstrapolacji

Ostateczna ewaluacja na rozłącznym zbiorze testowym przyniosła rozstrzygnięcia o fundamentalnym znaczeniu dla zrozumienia dynamiki skalowania modeli bajtowych. Poniższa tabela prezentuje zbiorcze zestawienie wyników dla wszystkich badanych konfiguracji w odniesieniu do klasycznych standardów kompresji.

**Metody klasyczne**

| Algorytm / Model | Liczba parametrów | Okno kontekstu | Test BPC |
|---|---|---|---|
| GZIP (DEFLATE) | nie dotyczy | 32 KB | 2,920 |
| bzip2 (BWT) | nie dotyczy | 900 KB | 2,200 |

**Modele neuronowe (architektura GPT-2 baseline)**

| Konfiguracja | Liczba parametrów | Okno kontekstu | Budżet kroków × batch | Test BPC |
|---|---|---|---|---|
| Medium (grid) | 3,4 M | 256 B | 2 000 × 16 | 4,080 |
| Large | 19,7 M | 1 024 B | 40 000 × 16 | 4,034 |
| XLarge | 57,8 M | 1 024 B | 20 000 × 8 | 3,987 |

**Modele neuronowe (architektura LLaMA-style)**

| Konfiguracja | Liczba parametrów | Okno kontekstu | Budżet kroków × batch | Test BPC |
|---|---|---|---|---|
| XLarge LLaMA | 57,0 M | 1 024 B | 20 000 × 8 | **3,920** |
| XXLarge LLaMA¹ | | 1 024 B | 5 200 × 8 (przerwany) | 3,879 |
| XXXLarge LLaMA | 202,9 M | 1 024 B | 40 000 × 4 | 3,956 |

¹ *Trening konfiguracji XXLarge LLaMA został przedwcześnie zakończony na 5 200 kroku z planowanych 20 000 ze względu na ograniczenia czasu obliczeniowego. Raportowana wartość pochodzi z najlepszego punktu kontrolnego w tym zakresie. Wynik nie jest zatem bezpośrednio porównywalny z pozostałymi konfiguracjami pełnoczasowymi i traktowany jest jako sygnał kierunkowy.*

*Tabela 5.2: Zestawienie końcowych wyników kompresji na zbiorze testowym enwik8. Wartości test BPC mierzone deterministycznie z wykorzystaniem rozłącznych okien kontekstowych.*

Najbardziej doniosłym odkryciem tej fazy eksperymentów jest **regresja wydajności dla największej w pełni wytrenowanej architektury**. Konfiguracja XXXLarge LLaMA, mimo ponad trzykrotnie większej pojemności strukturalnej niż XLarge LLaMA, osiągnęła na zbiorze testowym wynik 3,956 BPC, ulegając mniejszemu wariantowi (3,920 BPC). Zjawisko to obnaża drastyczne granice naiwnego skalowania parametrów w warunkach ściśle limitowanego budżetu danych treningowych.

Współczynnik stosunku przetworzonych tokenów do liczby parametrów (`D/N`) dla modelu XXXLarge wyniósł zaledwie **0,82**, podczas gdy optymalna wartość postulowana przez prawa Chinchilla oscyluje wokół **20**. Model o rozmiarze ponad 200 milionów wag znalazł się zatem w skrajnym reżimie niedotrenowania (`data starvation`). Ogromna pojemność sieci doprowadziła do nieefektywnej dystrybucji gradientów oraz spadku zdolności uogólniania względem mniejszej, lepiej nasyconej danymi jednostki XLarge LLaMA. Wynik ten falsyfikuje optymistyczne predykcje oparte na wczesnych krzywych uczenia i dowodzi, że w kompresji bajtowej zachowanie optymalnych proporcji między rozmiarem korpusu a kubaturą sieci jest warunkiem koniecznym do minimalizacji entropii.

Komplementarnym sygnałem jest wynik konfiguracji XXLarge LLaMA, która już po 5 200 kroku osiągnęła 3,879 BPC, co stanowi najniższą zaobserwowaną wartość spośród wszystkich badanych modeli. Mimo że trening tego wariantu nie został doprowadzony do końca, sam fakt szybkiego osiągnięcia wyniku lepszego od XLarge i XXXLarge przy tej samej procedurze treningowej sugeruje istnienie *sweet spotu* w przedziale 100–150 milionów parametrów dla rozważanego budżetu danych. Pełne przebadanie tego punktu architektonicznego stanowi naturalny kierunek dalszych prac.
