# 4. Implementacja eksperymentalna

Niniejszy rozdział szczegółowo opisuje proces transferu założeń teoretycznych do postaci funkcjonalnego oprogramowania badawczego. Kod źródłowy został zaimplementowany w systemowym języku Rust z wykorzystaniem niskopoziomowego frameworka tensorowego Candle. Zakres zrealizowanej implementacji obejmuje pełny potok treningu, ewaluacji oraz przeszukiwania przestrzeni hiperparametrów, umożliwiający rygorystyczną weryfikację metryki BPC dla różnych wariantów architektury Transformer. Pełen kod źródłowy wraz z konfiguracjami eksperymentów dostępny jest publicznie w repozytorium GitHub: https://github.com/Thomas-RC/thesis-compressor.

Świadomie ograniczono zakres implementacji do warstwy modelującej rozkład prawdopodobieństwa, traktując fizyczny koder entropijny oraz autoregresyjny dekoder jako moduły projektowe wykraczające poza zakres niniejszej pracy badawczej. Decyzja ta wynika z faktu, że metryka BPC, optymalizowana w trakcie treningu, stanowi ścisłą teoretyczną granicę kompresji wyznaczoną przez twierdzenie Shannona o kodowaniu źródła. Skuteczność architektury jako estymatora rozkładu może być zatem zwalidowana niezależnie od warstwy fizycznej kompresji, co zostało omówione szczegółowo w podrozdziale 4.6.

## 4.1. Pipeline danych i reprezentacja byte-level

Pierwszym etapem budowy systemu było zaprojektowanie wydajnego potoku zasilającego model danymi uczącymi. W przeciwieństwie do standardowych modeli przetwarzania języka naturalnego, opartych na złożonych tokenizatorach pod-słownych, proponowany system operuje bezpośrednio na surowych strumieniach bajtów. Korpus enwik8 wczytywany jest jednorazowo do pamięci operacyjnej jako jednowymiarowa tablica wartości 8-bitowych bez znaku (typ `u8`), a następnie deterministycznie dzielony na trzy rozłączne podzbiory: treningowy (90 MB), walidacyjny (5 MB) oraz testowy (5 MB).

Zasilanie sieci neuronowej realizowane jest z wykorzystaniem stochastycznego generatora przesuwnego okna. Dla każdej próbki w przygotowywanej paczce danych losowany jest indeks początkowy, z którego wycinany jest ciąg bajtów o długości odpowiadającej zdefiniowanemu oknu kontekstowemu. Równolegle tworzony jest wektor docelowych etykiet, przesunięty względem wejścia dokładnie o jedną pozycję w prawo. Takie podejście gwarantuje, że proces uczenia ma charakter ściśle autoregresyjny, a zadaniem sieci jest estymacja rozkładu warunkowego dla każdego kolejnego elementu sekwencji.

Z uwagi na rygorystyczne wymagania typowania biblioteki Candle, surowe bajty przed przekazaniem do warstwy osadzeń są bezstratnie rzutowane na 32-bitowe liczby całkowite. Zrezygnowanie z zewnętrznych bibliotek przetwarzania tekstu pozwoliło na całkowite wyeliminowanie narzutu obliczeniowego na etapie przygotowania danych — proces ten przebiega niemal bezkosztowo z punktu widzenia cykli procesora głównego, co umożliwia pełne wysycenie układu graficznego operacjami tensorowymi.

## 4.2. Implementacja modelu i optymalizacje obliczeniowe

Proces budowy architektury w środowisku Candle wymagał precyzyjnej kontroli nad układem oraz alokacją tensorów. Pierwszym z kluczowych wyzwań była optymalizacja wielogłowicowego mechanizmu atencji. W klasycznym ujęciu teoretycznym zapytania (Q), klucze (K) oraz wartości (V) obliczane są za pomocą trzech niezależnych warstw liniowych. W proponowanej implementacji zastosowano technikę fuzji operacji, gdzie wszystkie trzy projekcje realizowane są przez pojedynczą, trzykrotnie szerszą macierz wag o wymiarach `d_model × 3·d_model`. Wynik tego mnożenia macierzowego jest następnie dynamicznie dzielony w locie za pomocą operacji porcjowania (`chunk(3, 2)` w terminologii Candle). Takie podejście znacząco redukuje narzut związany z alokacją pamięci podręcznej oraz z wywoływaniem osobnych kerneli obliczeniowych na układzie graficznym, co bezpośrednio przekłada się na przepustowość treningu.

Drugim istotnym elementem optymalizacyjnym była implementacja maskowania przyczynowego, niezbędnego w architekturach dekoderowych. Zamiast generować trójkątną maskę w każdym kroku pętli uczącej, maska dla maksymalnej długości kontekstu jest alokowana jednorazowo podczas inicjalizacji modelu i wypełniana wartościami ujemnej nieskończoności w górnym trójkącie. W trakcie fazy propagacji w przód pobierany jest z niej jedynie wycinek o wymiarach odpowiadających aktualnej sekwencji, który następnie poddawany jest operacji rozgłaszania (broadcasting) do wszystkich głów atencji przed aplikacją funkcji softmax. Ze względu na rygorystyczne zarządzanie pamięcią w języku Rust, konieczne było jawne wymuszanie ciągłości pamięci tensorów (operacja `.contiguous()`) po każdej modyfikacji ich kształtu. Brak tej operacji prowadził do błędów wykonania związanych z fragmentacją wskaźników wewnątrz frameworka.

## 4.3. Modularna architektura systemu i warianty modelu

W celu przeprowadzenia rygorystycznej analizy ablacyjnej zaprojektowano modularną architekturę kodu, w której warstwa abstrakcji `Arch` umożliwia uruchamianie eksperymentów z dowolnym wariantem modelu bez konieczności modyfikacji potoku treningowego ani ewaluacyjnego. Każdy z wariantów został zaimplementowany w osobnym module źródłowym, dzielącym wspólny interfejs propagacji w przód.

Zaimplementowano pięć wariantów architektury, różniących się wewnętrznymi mechanizmami matematycznymi:

1. **Baseline (`model.rs`)** — referencyjna architektura w stylu GPT-2, wykorzystująca wyuczalne osadzenia pozycyjne, normalizację warstwy (LayerNorm) z parametrem przesunięcia oraz nieliniową funkcję aktywacji GELU w sieci jednokierunkowej.
2. **RoPE (`model_rope.rs`)** — wariant zastępujący wyuczalne osadzenia pozycyjne mechanizmem obrotowego kodowania pozycji (Rotary Positional Embeddings), aplikującym macierze rotacji do wektorów zapytań i kluczy przed obliczeniem iloczynu skalarnego.
3. **RMSNorm (`model_rms.rs`)** — wariant z uproszczoną normalizacją Root Mean Square, pomijającą obliczanie wartości średniej oraz parametru przesunięcia.
4. **SwiGLU (`model_swiglu.rs`)** — wariant z bramkowaną funkcją aktywacji w sieci jednokierunkowej, gdzie wyjście warstwy liniowej jest mnożone elementowo przez aktywację Swish drugiej projekcji.
5. **LLaMA-full (`model_llama.rs`)** — pełna integracja wszystkich powyższych komponentów, odzwierciedlająca topologię modelu LLaMA z usuniętymi parametrami przesunięcia w warstwach liniowych.

Dla mechanizmu RoPE tablice współczynników cosinusów oraz sinusów obliczane są jednorazowo podczas inicjalizacji modelu i przechowywane w pamięci urządzenia jako stałe tensory. Eliminuje to konieczność powtórnego ich generowania w każdym kroku treningu, co byłoby istotnym wąskim gardłem w pętli optymalizacyjnej. Konfiguracja modelu (struktura `Config`) udostępnia cztery predefiniowane presety o rosnącej pojemności: `small` (0,5 mln parametrów), `medium` (3,4 mln), `large` (19,7 mln) oraz `xlarge` (57,8 mln), umożliwiające szybkie przełączanie między skalami eksperymentu.

## 4.4. Harmonogram uczenia i procedura treningowa

Zapewnienie stabilnej zbieżności modelu na wczesnych etapach optymalizacji wymagało implementacji autorskiego harmonogramu współczynnika uczenia. Zamiast wartości stałej zastosowano funkcję ciągłą złożoną z trzech faz: liniowej rozgrzewki, stabilnego plateau oraz liniowego wygaszania. Przebieg ten opisuje wzór:

$$\eta(t) = \begin{cases} \eta_{min} + t \cdot \frac{\eta_{max} - \eta_{min}}{t_{warmup}} & \text{dla } t \le t_{warmup} \\ \eta_{max} & \text{dla } t_{warmup} < t \le t_{decay} \\ \eta_{max} - (\eta_{max} - \eta_{min}) \cdot \frac{t - t_{decay}}{T_{max} - t_{decay}} & \text{dla } t > t_{decay} \end{cases}$$

gdzie `T_max` oznacza całkowitą zaplanowaną liczbę kroków treningowych, `t_warmup` długość fazy rozgrzewki pozwalającą uniknąć destabilizacji gradientów na początku treningu, a `t_decay` wyznacza moment rozpoczęcia wygaszania. W zaimplementowanej procedurze faza wygaszania uruchamiana jest po osiągnięciu 80% całkowitego budżetu kroków (parametr `decay_start_frac = 0.8`), a długość rozgrzewki ustawiona jest domyślnie na 25 kroków. Optymalizator AdamW operuje ze stałą regularyzacją wagową (`weight_decay = 0.01`), a wartości momentów β₁ i β₂ przyjęto zgodnie z konwencją PyTorcha.

Funkcja kosztu opiera się na entropii krzyżowej obliczanej na surowych logitach dla 256 klas bajtowych. Z uwagi na fakt, iż implementacja w bibliotece Candle bazuje na logarytmie naturalnym, wynikowa wartość straty wyrażona jest w natach. Konwersja na docelową metrykę BPC zrealizowana jest poprzez dzielenie przez stałą `ln(2)`, zgodnie z teoretycznym uzasadnieniem przedstawionym w rozdziale 3.4.

## 4.5. Ewaluacja, automatyzacja i interfejs CLI

Zasadniczym etapem weryfikacji poprawności matematycznej zaimplementowanej architektury było przeprowadzenie pierwszej ewaluacji typu MVP. Przed wdrożeniem modelu do środowiska akcelerowanego sprzętowo wykonano cykl 500 kroków optymalizacyjnych na procesorze głównym przy użyciu konfiguracji `small` o rozmiarze około 0,49 miliona parametrów. Eksperyment ten potwierdził poprawność wstecznej propagacji błędu oraz integracji harmonogramu współczynnika uczenia z optymalizatorem. W kroku początkowym model wykazywał wartość BPC na poziomie 8,47, czyli zbliżonym do losowej predykcji w słowniku 256-elementowym, a wraz z postępem treningu metryka spadała monotonicznie do 4,88 BPC po 500 krokach. Stabilność tego procesu dopuściła kod do wdrożenia we właściwym środowisku akcelerowanym kartą graficzną.

| Krok | Train Loss (nat) | Train BPC | Valid BPC |
|------|------------------|-----------|-----------|
| 1    | 5,87             | 8,47      | —         |
| 50   | 5,11             | 7,37      | 7,36      |
| 100  | 4,51             | 6,51      | 6,54      |
| 300  | 3,67             | 5,30      | 5,27      |
| 500  | 3,49             | 5,03      | 4,88      |

*Tabela 4.1: Walidacja MVP — przebieg 500 kroków treningu na CPU dla konfiguracji `small`.*

W celu zapewnienia powtarzalności środowiska badawczego zrezygnowano z twardego kodowania parametrów na rzecz dynamicznego interfejsu wiersza poleceń zaimplementowanego za pomocą biblioteki `clap`. Pozwala to na wstrzykiwanie hiperparametrów (rozmiar modelu, długość okna, liczba kroków, wariant architektury) bez konieczności każdorazowej rekompilacji kodu. System wzbogacono o moduł automatycznego przeszukiwania przestrzeni hiperparametrów (`grid.rs`), który iteruje po zdefiniowanej siatce wymiarów osadzeń, liczby warstw oraz długości kontekstu, automatycznie agregując wyniki do pliku CSV stanowiącego fundament dla analizy przedstawionej w rozdziale 5.

Dedykowana binarka ewaluacyjna (`eval.rs`) realizuje deterministyczną procedurę pomiarową na pełnym zbiorze testowym z wykorzystaniem rozłącznych okien kontekstowych, dzięki czemu raportowane wartości BPC są wolne od szumu statystycznego wynikającego z losowego doboru paczek. Procedura ta jest wywoływana automatycznie po zakończeniu treningu długoterminowego, co eliminuje ryzyko niespójności metodologicznej między pomiarami treningowymi a końcowymi.

## 4.6. Projekt kompresora: złożoność dekodowania i integracja z koderem entropijnym

Zakres implementacji niniejszej pracy ograniczono do warstwy modelującej rozkład prawdopodobieństwa, traktując pełen kompresor jako naturalne rozszerzenie wykraczające poza obszar badawczy. Decyzja ta wynika z dwóch strukturalnych wyzwań inżynierskich, których analiza teoretyczna jest istotnym wkładem niniejszego rozdziału.

**Złożoność obliczeniowa autoregresyjnego dekodowania.** Podczas gdy faza treningu i ewaluacji może być realizowana równolegle dla całego okna kontekstowego, dekompresja stanowi fundamentalnie odmienny problem. Proces dekompresji w architekturze autoregresyjnej polegałby na generowaniu każdego kolejnego bajtu na podstawie bajtów już zdekodowanych. Naiwna implementacja wymagałaby ponownego przeliczenia całego stanu uwagi dla całej dotychczasowej sekwencji przy każdym nowym znaku, co prowadzi do złożoności O(N²) względem długości pliku. W kontekście strumieni liczonych w dziesiątkach megabajtów takie podejście uniemożliwiałoby praktyczne zastosowanie narzędzia.

Standardowym rozwiązaniem tego problemu jest mechanizm buforowania kluczy i wartości (KV-Cache), polegający na utrzymywaniu w pamięci wektorów K i V z poprzednich kroków czasowych. W momencie predykcji bajtu na pozycji `i` model obliczałby jedynie wektor zapytania dla bieżącego kroku, a iloczyn skalarny wykonywałby z buforem wektorów historycznych. Złożoność generacji pojedynczego znaku redukuje się wówczas do O(1), a całkowity czas dekompresji staje się liniowy O(N). Pełna implementacja KV-Cache w środowisku Candle wymaga jednak dynamicznego zarządzania alokacjami pamięci urządzenia oraz obsługi specyficznych przypadków brzegowych przy przekraczaniu maksymalnej długości kontekstu, co kwalifikuje to jako kierunek dalszych prac inżynierskich.

**Integracja z koderem entropijnym.** Drugim elementem dopełniającym praktyczny kompresor jest fizyczny koder entropijny, taki jak Asymmetric Numeral Systems (ANS). Sprzężenie wyjścia sieci z koderem polegałoby na konsumowaniu rozkładu prawdopodobieństwa generowanego przez model i przypisywaniu aktualnemu bajtowi kodu binarnego o długości zbliżonej do informacji własnej `−log₂ P(xᵢ | x<ᵢ)`. Bajty o wysokim prawdopodobieństwie predykcji byłyby wówczas zapisywane przy użyciu ułamkowej liczby bitów, co pozwala na maksymalne zbliżenie się do teoretycznej granicy entropii.

**Projektowany format bitstreamu.** Zaprojektowany format pliku skompresowanego składałby się z dwóch integralnych sekcji. Pierwszą stanowiłby znormalizowany nagłówek zawierający strukturę grafu obliczeniowego oraz pełny zrzut wag modelu w formacie `safetensors`. Stanowi to niezbędny narzut pamięciowy, który musi zostać dostarczony do dekompresora w celu odtworzenia identycznego stanu predykcyjnego sieci. Drugą sekcję stanowiłby właściwy strumień danych wygenerowany przez koder ANS. Tak zdefiniowany format wymusza architektoniczne uwzględnienie reguły minimalnej długości opisu (MDL): praktyczna efektywność kompresji dla konkretnego pliku zależy od tego, czy oszczędności uzyskane na poziomie ułamkowych bitów z kodera entropijnego pokryją z nawiązką stały koszt dołączenia wag sieci do nagłówka. Dla małych plików narzut ten może zniwelować zysk z precyzyjnej predykcji, natomiast dla korpusów rzędu dziesiątek megabajtów dominującym czynnikiem staje się jakość samego modelu probabilistycznego — co stanowi główny przedmiot badań niniejszej pracy.
