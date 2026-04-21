## Water table and water draw measurements

## How to extract well data?
Script well_data.py poskytuje 3 základní funkce pro zpracování dat:
- read_sections: Funkce načte a zpracuje základní data vrtů a jejich sections. Data jsou načítána z jednoho souboru a jednoho sheetname. Výstupem je pandas.DataFrame. Funkce nezpracovává data měření.
- read_sections_water_levels: Funkce načte a zpracuje data měření hladin vrtů. Funkce umožňuje zpracovat data z jednoho nebo více vstupních souborů, data každého vrtu jsou obsažena na samostatném sheetname. Zpracování jednotlivých sheetname je prováděno automaticky. Ke každému vrtu je automaticky doplněno označení vrtu v souboru sections (viz. předchozí funkce). Zpracovaná data jsou uložena do zar_fuse úložiště, pomocným výstupem, který funkce vrací, je pandas.DataFrame. 
- read_draw: Funkce načte a zpracuje data čerpání z vodního zdroje pro jeden vrt. Data jsou načítána z jednoho souboru a jednoho sheetname. Výstup je ukládán do zar_fuse úložiště, pomocným výstupem, který funkce vrací, je pandas.DataFrame.
Formát tabulek a jejich sloupců je popsán níže.
For correct access to zarr_fuse storage we need to have .secrets_env file defined in root directory of project. This file defines access keys to zarr_fuse storage.

Příklad použití všech výše popsaných funkcí je v testu /tests/ingress/well_data/test_borehole_water_level.py.

## File overview
- `1.CS_Zpráva o vlivu na životní prostředí.pdf` - polská zpráva EIA o rozšíření těžby v dole Turow, obsahuje i predikce jejich HG modelu
- `25_09_27_Odbery_Uhelna.xlsx`         - čerpání z vodního zdroje Uhelná, řádky roky, sloupce Mx čerpané objemy po měsících, výšky hladin
- 25_09_27_vrty_III.etapa_vše.xlsx    - výšky hladin na monitorovacích vrtech, klíčový je sloupec FINAL
- 25_09_27_vrty_nové_vše.xlsx
- 25_09_27_vrty_staré_vše.xlsx
- struktura_dat_Odberu.xls            - Plné popisy sloupců z tabulky: 25_09_27_Odbery_Uhelna.xlsx
- Vrty_souradnice_perforace.xlsx      - Přehled poloh zhlaví X,Y,Z a pažení vrtů

TODO: Python skript pro čtení tabulek a jejich případné ruční úpravy

- výsledný formát odběry: 
  pandas DataFrame se sloupci: 'date' (type: datetime64[day]), 'cum_draw' (type: float, unit:m3)
  'date' je 28 den sledovaného měsíce
  
- výsledný formát hladiny vrtů:
  pandas dataframe, sloupce: 'date_time' (datetime64[min]), 'well_id' (str), 'water_table' (float [m])
  Ignorovat vrty z daného seznamu 'well_id', vyrobit senzam podezřelých. Dvojice Soubor, well id.
  
- tabulka s přehledem vrtů:
    pandas DF, sloupce: 'well_id' (str), 'X', 'Y', 'Z' (použít sloupec ZOB), 
                        'depth' (float), 'collector' (str),
                        'interval_min', 'interval_max' (float)
                        'interval_num_from_top' (0, 1, ...) 
                        'interval_max'
    
    test konzistence Z_OD = ZOB - DO; Z_DO = ZOB - OD
    test konzistence interval_min (min_id) ...až intevarval_max (max_id)  = OD - DO
                        
                        
                        
    pokud některý chybí hodnota NaN
    'collector' je podle sloupce 'Kolektor_REV'
    Operativně XLSX tabulku upravit (doplnit chybějící hodnoty Perf-dílci, podle OD, DO
    

    matplotlib, grafy průběhů odběru a hladin -> PDF
    
## Formát tabulek generovaných funkcemi skriptu well_data.py

- sloupce generované funkcí read_water_level
  - date_time               datum a čas měření
  - well_id                 id vrtu
  - water_level             nadmořská výška hladiny vody [m nm]
  
- sloupce generované funkcí read_sections
  - well_id                 id vrtu
  - borehole_full_name      celé jméno vrtu ve vstupním souboru (včetně názvu kolektoru)
  - X                       x-souřadnice vrtu [m]
  - Y                       y-souřadnice vrtu [m]
  - Z                       z-souřadnice vrtu [m]
  - collector               název kolektoru [m]
  - depth                   hloubka vrtu [m]
  - sheetname_in_water_file název tabulky v souboru měření hladin vrtů
  - confirmed               indikátor existující tabulky v souboru měření vrtů
  - interval_max            maximum měřeného intervalu [m]
  - interval_min            minimum měřeného intervalu [m]
  - interval_num_from_top   index měřeného intervalu v rámci měření stejného vrtu (měření seřazena vzestupně od povrchu))

- sloupce generované funkcí read_draw
  - date                    datum a čas měření
  - cum_draw                množství odčerpané vody za kalendářní měsích [m^3]
  - well_id                 id vrtu
  
## Přehled možných errors a warnings

- ERROR: Chybové hlášky 'AssertionError: Session was never entered', 'HTTPClientError' apod. vznikají při ukončení spojení se zarr repository
- WARNING: 'Could not infer format ...' vygeneruje pandas při detekci neplatného formátu dat v sheetname ve vstupním souboru (zpravidla se jedná o vysvětlivky apod.). Následně je vypsána hláška o přeskočení dané tabulky.
- WARNING: 'Invalid value ...' zobrazují drobné nekonzistence v souboru s přehledem vrtů. Jedná se o sloupce, které přímo nesouvisí se zpracováním dat a tudíž nemají vliv na výpočet.
- WARNING: "ZarrUserWarning" apod. generuje package Zarr.
