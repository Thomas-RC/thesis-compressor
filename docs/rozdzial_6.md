# 6. Analiza ablacyjna i dekompozycja zysku informacyjnego

W niniejszym rozdziale poddano szczegółowej analizie wpływ poszczególnych komponentów architektury LLaMA na ostateczną wydajność kompresji. Celem badania było ilościowe wyizolowanie efektu wnoszonego przez każdą z trzech kluczowych modyfikacji matematycznych względem bazowego modelu typu GPT-2: zastąpienia LayerNorm przez RMSNorm, wyuczalnych osadzeń pozycyjnych przez obrotowe kodowanie pozycji (RoPE) oraz funkcji aktywacji GELU przez bramkowaną aktywację SwiGLU. Wyniki rzucają nowe światło na naturę interakcji między mechanizmami normalizacji, kodowania pozycji oraz nieliniowości w reżimie modelowania na poziomie bajtów.

## 6.1. Procedura ablacyjna i kontrola zmiennych

Aby uzyskać porównywalne i statystycznie wiarygodne wyniki, każdy z eksperymentów ablacyjnych przeprowadzono z zachowaniem ścisłej kontroli zmiennych. Wszystkie pięć konfiguracji (baseline GPT-2, baseline + RMSNorm, baseline + RoPE, baseline + SwiGLU, pełny pakiet LLaMA) trenowano w **identycznej procedurze**: preset `xlarge` (~57 mln parametrów, `d_model = 768`, `n_layers = 8`, `n_heads = 8`), długość okna kontekstowego 1024 bajty, 20 000 kroków optymalizatora przy `batch_size = 8`, harmonogram współczynnika uczenia z 200-krokową rozgrzewką oraz fazą wygaszania uruchamianą po osiągnięciu 80% budżetu kroków. Łączny budżet danych treningowych wyniósł zatem 163 840 000 tokenów dla każdego z wariantów. Tak ustawione warunki gwarantują, że obserwowane różnice w metryce BPC nie wynikają z rozbieżności w procedurze treningowej, lecz wyłącznie ze zmian w samej topologii sieci.

Każda konfiguracja została następnie poddana deterministycznej ewaluacji końcowej na rozłącznym zbiorze testowym (5 MB), z wykorzystaniem rozłącznych okien kontekstowych. Skrypty ablacyjne (`scripts/run_ablations.sh`) zostały uruchomione sekwencyjnie po zakończeniu wcześniejszego treningu, aby uniknąć rywalizacji o pamięć VRAM. Wszystkie czterocyfrowe wartości test BPC raportowane w niniejszym rozdziale pochodzą bezpośrednio z artefaktu `results/eval_summary.csv` i są w pełni reprodukowalne na publicznie udostępnionym kodzie źródłowym.

## 6.2. Izolacja komponentów i dominacja mechanizmu RoPE

Badanie rozpoczęto od wprowadzenia pojedynczych zmian do konfiguracji XLarge przy zachowaniu pozostałych parametrów modelu bazowego. Wyniki zestawione w poniższej tabeli ujawniły nieoczekiwaną i mocno asymetryczną strukturę zysków oraz strat poszczególnych komponentów.

| Modyfikacja (Ablacja) | Test BPC | Δ vs baseline | Interpretacja |
|---|---|---|---|
| Baseline (XLarge GPT-2) | 3,987 | 0,000 | Punkt referencyjny |
| Tylko RMSNorm | 3,987 | −0,000 | Neutralność jakościowa |
| Tylko RoPE | 3,818 | **−0,169** | Dominujący czynnik poprawy |
| Tylko SwiGLU | 4,205 | **+0,218** | Znacząca regresja |
| Pełny pakiet LLaMA | 3,920 | −0,067 | Efekt synergiczny |

*Tabela 6.1: Wyniki ablacji jednoczynnikowej dla architektury XLarge (~57 mln parametrów, 20 000 kroków, batch 8). Wartości test BPC pochodzą z deterministycznej ewaluacji na pełnym zbiorze testowym enwik8.*

Najsilniejszym czynnikiem determinującym spadek entropii okazało się **obrotowe kodowanie pozycji** (Rotary Positional Embeddings). Pojedyncze wdrożenie tego komponentu, przy zachowaniu wszystkich pozostałych elementów architektury bazowej, obniżyło błąd modelu z 3,987 do 3,818 BPC, co stanowi redukcję o 0,169 BPC. RoPE umożliwiło mechanizmowi atencji znacznie skuteczniejsze modelowanie relatywnych odległości między bajtami w porównaniu do klasycznych wyuczalnych osadzeń pozycyjnych. Zysk uzyskany wyłącznie przez tę zmianę sugeruje, że precyzyjna informacja o relacjach przestrzennych jest dla skuteczności kompresji ustrukturyzowanych danych tekstowych ważniejsza niż głębokość samej sieci czy rodzaj zastosowanej normalizacji.

Wynik ten ma silne uzasadnienie teoretyczne. W odróżnieniu od osadzeń wyuczalnych, które wymagają od modelu samodzielnego odkrycia struktury translacyjnej w przestrzeni reprezentacji, RoPE w sposób matematycznie ścisły koduje informację o relatywnym przesunięciu między pozycjami poprzez rotację wektorów zapytań i kluczy. W kontekście danych typu enwik8, zawierających powtarzalne struktury składniowe XML, znaczniki MediaWiki oraz tagi hipertekstowe, zdolność do uchwycenia odległościowo-relatywnych wzorców okazuje się szczególnie cenna.

Z kolei zastąpienie warstwy LayerNorm przez RMSNorm okazało się zmianą **jakościowo neutralną** — różnica wyniku (−0,000242 BPC) jest pomijalna i mieści się w zakresie szumu pomiarowego. Potwierdza to, że RMSNorm w izolacji nie stanowi samodzielnego źródła poprawy modelu, lecz raczej upraszczającą modyfikację implementacyjną kompatybilną z dynamiką gradientów wymaganą przez pozostałe komponenty pakietu LLaMA.

## 6.3. Paradoks funkcji SwiGLU i błędy izolacji

Najbardziej zaskakującym wynikiem analizy ablacyjnej okazała się gwałtowna regresja wydajności po wprowadzeniu samej funkcji aktywacji SwiGLU. Model odnotował wzrost błędu o 0,218 BPC (z 3,987 do 4,205), co w naiwnej interpretacji sugerowałoby konieczność odrzucenia tego komponentu jako szkodliwego dla architektury. Pełna analiza ujawnia jednak głębsze przyczyny tego zjawiska, które są kluczowe dla zrozumienia natury nowoczesnych architektur transformacyjnych.

SwiGLU jako mechanizm bramkowany wymaga specyficznej dynamiki gradientów oraz odpowiedniego skalowania pośrednich aktywacji, których standardowa normalizacja LayerNorm wraz z klasycznym schematem inicjalizacji wag nie są w stanie zapewnić. W szczególności, bramka multiplikatywna w SwiGLU wprowadza nieliniową interakcję między dwiema niezależnymi projekcjami liniowymi, co istotnie zmienia rozkład wartości w tensorze pośrednim. Bez kompensacji ze strony bardziej restrykcyjnej normalizacji RMSNorm dochodzi do destabilizacji propagacji sygnału w głębszych warstwach sieci.

Drugim prawdopodobnym czynnikiem niepowodzenia w izolacji jest pozostawienie parametrów przesunięcia (`bias`) w warstwach liniowych — w oryginalnej specyfikacji LLaMA wszystkie warstwy gęste są bezbiasowe, co stanowi celową regularyzację przepływu informacji przez bramki aktywacji. Eksperyment SwiGLU-w-izolacji łamie tę regułę, ponieważ szkielet bazowy GPT-2 zachowuje parametry przesunięcia zgodnie z konwencją oryginalnej publikacji.

Wynik ten ma istotne znaczenie metodologiczne. Pokazuje on, że ablacje jednoczynnikowe — choć metodologicznie poprawne i obowiązkowe w rzetelnej analizie — mogą prowadzić do **systematycznie błędnych wniosków inżynierskich**, jeśli ocenia się je w oderwaniu od pełnego ekosystemu komponentów współdziałających ze sobą. SwiGLU sam w sobie nie jest wadliwą funkcją aktywacji; jego efektywność jest jednak warunkowana obecnością odpowiednich elementów towarzyszących.

## 6.4. Efekt synergii: całość większa niż suma części

Kluczowym odkryciem badawczym jest empiryczne wykazanie silnej synergii między testowanymi modułami architektury LLaMA. Sumaryczny efekt cząstkowy wszystkich trzech zmian, mierzony poprzez ich całkowicie niezależną izolację, można zsumować algebraicznie:

$$\Delta_{\text{cząstkowy}} = \Delta_{\text{RMSNorm}} + \Delta_{\text{RoPE}} + \Delta_{\text{SwiGLU}} = -0{,}000 + (-0{,}169) + (+0{,}218) = +0{,}049 \text{ BPC}$$

Naiwna ekstrapolacja sugerowałaby zatem, że pełna integracja wszystkich komponentów powinna pogorszyć wynik bazowy o 0,049 BPC, prowadząc do błędu rzędu **4,036 BPC**. Tymczasem rzeczywisty pomiar pełnego pakietu LLaMA daje wynik **3,920 BPC**, co stanowi redukcję o 0,067 BPC w stosunku do architektury bazowej.

Różnica między czysto teoretycznym wynikiem wynikającym z dodania pojedynczych efektów ablacyjnych (4,036 BPC) a rzeczywistym wynikiem wariantu zintegrowanego (3,920 BPC) wynosi aż **0,116 BPC**. Wartość ta — przewyższająca największy pojedynczy wkład ablacyjny (RoPE, 0,169 BPC) — stanowi empiryczny dowód na istnienie **silnego efektu synergicznego**. Komponenty architektury LLaMA nie są niezależnymi optymalizacjami addytywnymi, lecz stanowią matematycznie sprzężony układ, w którym każdy element kompensuje niedoskonałości pozostałych:

- **RMSNorm** dostarcza odpowiednią dynamikę normalizacji, która stabilizuje wyjście warstwy SwiGLU,
- **RoPE** zapewnia precyzyjny sygnał relacyjny, który umożliwia atencji efektywne wykorzystanie wzbogaconej reprezentacji,
- **SwiGLU** przy obecności RMSNorm i braku biasów efektywnie modeluje nieliniowe cechy bez destabilizacji gradientów.

Wzajemna komplementarność tych mechanizmów oznacza, że żaden z nich nie powinien być rozważany w oderwaniu od pozostałych. To z kolei jest istotnym ostrzeżeniem dla projektantów systemów uczących się: tradycyjna metodyka analizy jednoczynnikowej, choć metodologicznie czysta, może prowadzić do mylnych wniosków przy projektowaniu nowoczesnych głębokich sieci neuronowych.

## 6.5. Synteza odkryć architektonicznych i implikacje praktyczne

Przeprowadzone analizy ablacyjne oraz weryfikacja empiryczna gigantycznych architektur (zaprezentowana w rozdziale 5) pozwalają na sformułowanie trzech fundamentalnych wniosków naukowych, które stanowią główną tezę niniejszej dysertacji.

**1. Dominacja obrotowego kodowania pozycji (RoPE).** Najsilniejszym pojedynczym czynnikiem optymalizacyjnym w nowoczesnych architekturach transformacyjnych operujących na poziomie bajtów jest mechanizm RoPE. Wyizolowane wdrożenie tego komponentu obniżyło błąd modelu bazowego o 0,169 BPC — wartość przewyższającą sumaryczny zysk pełnego pakietu LLaMA mierzony względem GPT-2. Udowadnia to, że precyzyjne mapowanie relatywnych odległości między znakami jest dla skuteczności kompresji ważniejsze niż głębokość sieci czy rodzaj zastosowanej normalizacji. Z perspektywy praktycznej oznacza to, że pierwszą rekomendowaną modyfikacją dla każdej istniejącej architektury byte-level powinno być właśnie wdrożenie RoPE, jako interwencja o najwyższym stosunku zysku do złożoności implementacyjnej.

**2. Paradoks regresji i synergia mechanizmu SwiGLU.** Badania udowodniły, że nowoczesne funkcje aktywacji nie stanowią uniwersalnych rozwiązań gwarantujących natychmiastową poprawę jakości. Implementacja wyłącznie mechanizmu SwiGLU w środowisku niedostosowanym (pozostawienie parametrów przesunięcia w warstwach liniowych oraz brak odpowiedniej normalizacji RMSNorm) wywołała gwałtowną regresję modelu o 0,218 BPC. Jednocześnie pełna integracja komponentów LLaMA przyniosła wynik o 0,116 BPC lepszy niż suma wyizolowanych efektów cząstkowych. Zjawisko to dowodzi, że architektury neuronowe stanowią nierozerwalne ekosystemy matematyczne, których elementy wchodzą w silne, nieliniowe interakcje. Implikacja praktyczna: refaktoryzacja istniejących architektur powinna być realizowana **w pełnych pakietach**, a nie pojedynczymi komponentami.

**3. Szkodliwość naiwnego skalowania parametrów.** Eksperyment z modelem XXXLarge (zaprezentowany szczegółowo w rozdziale 5.6) dostarczył twardego dowodu na to, że w reżimie ograniczonego budżetu danych powiększanie struktury wag przynosi efekty przeciwne do zamierzonych. Model o rozmiarze ponad 200 milionów parametrów osiągnął wynik gorszy od jednostki trzykrotnie mniejszej, znajdując się w skrajnym reżimie niedotrenowania (`D/N ≈ 0,82` wobec optymalnego `≈ 20`). Falsyfikuje to powszechne w inżynierii uczenia maszynowego założenie o bezwzględnej wyższości potężniejszych modeli i podkreśla konieczność rygorystycznego bilansowania skali parametrów ze skalą dostępnego korpusu w ścisłej zgodności z empirycznymi prawami skalowania. Dla przyszłych badań w domenie kompresji bajtowej oznacza to, że zwiększanie pojemności sieci powinno być zawsze poprzedzone zwiększeniem objętości danych treningowych — w przeciwnym razie inwestycja obliczeniowa skutkuje pogorszeniem wyniku końcowego.

Powyższe trzy wnioski razem tworzą spójny obraz natury nowoczesnych modeli kompresji byte-level: efektywność architektury Transformer jako estymatora rozkładu prawdopodobieństwa zależy nie tyle od jej skali, co od **strukturalnej spójności zastosowanych mechanizmów matematycznych** oraz od zachowania właściwych proporcji między pojemnością modelu a budżetem danych.
