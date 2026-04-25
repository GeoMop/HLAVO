# Water table and water draw measurements

This package updates store of  `wells_schema.yaml`, exposed in the project schema index as
`hlavo/schemas/wells_schema.yaml`.

## Schema Update Procedure

1. Pull or refresh the source XLSX files with DVC.
2. Run the processing script from this directory through the project container
   wrapper:

```commandline
python well_data_process.py
```

3. To also generate a single multi-page PDF with water draw first and then
   water levels after writing to zarr, run:

```commandline
python well_data_process.py plot
```

`well_data_process.py` reads the Excel files, removes the previous target
zarr-fuse store, writes `Uhelna/water_draw`, writes `Uhelna/water_levels`, and
optionally plots the written data. The default S3 zarr-fuse store specified in
the schema requires credentials in `.secrets_env`; see the main README.md.

## Tests

Use the project test wrapper from `tests/`:

```bash
PATH=/home/hlavo/workspace/dev/venv-docker/bin:$PATH \
PYTEST_ADDOPTS="ingress/well_data/test_well_data.py" \
bash ./run
```

For a quick local-store smoke test, run the two lighter tests:

```bash
PATH=/home/hlavo/workspace/dev/venv-docker/bin:$PATH \
PYTEST_ADDOPTS="ingress/well_data/test_well_data.py::test_borehole_sections ingress/well_data/test_well_data.py::test_borehole_draw" \
bash ./run
```

## File overview
- `1.CS_Zpráva o vlivu na životní prostředí.pdf` - polská zpráva EIA o rozšíření těžby v dole Turow, obsahuje i predikce jejich HG modelu
- `25_09_27_Odbery_Uhelna.xlsx`         - čerpání z vodního zdroje Uhelná, řádky roky, sloupce Mx čerpané objemy po měsících, výšky hladin
- 25_09_27_vrty_III.etapa_vše.xlsx    - výšky hladin na monitorovacích vrtech, klíčový je sloupec FINAL
- 25_09_27_vrty_nové_vše.xlsx
- 25_09_27_vrty_staré_vše.xlsx
- struktura_dat_Odberu.xls            - Plné popisy sloupců z tabulky: 25_09_27_Odbery_Uhelna.xlsx
- Vrty_souradnice_perforace.xlsx      - Přehled poloh zhlaví X,Y,Z a pažení vrtů

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
