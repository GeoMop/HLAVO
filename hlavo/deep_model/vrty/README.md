## Water table and water draw measurements

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
    
    test konzistence Z_OD = ZOB - DO; Z_DO = ZOB - OD
    test konzistence interval_min (min_id) ...až intevarval_max (max_id)  = OD - DO
                        
                        
                        
    pokud některý chybí hodnota NaN
    'collector' je podle sloupce 'Kolektor_REV'
    Operativně XLSX tabulku upravit (doplnit chybějící hodnoty Perf-dílci, podle OD, DO
    

    matplotlib, grafy průběhů odběru a hladin -> PDF
    
