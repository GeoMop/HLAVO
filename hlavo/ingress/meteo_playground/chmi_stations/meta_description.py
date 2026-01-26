
# stations
meta1 = {
    "WSI": (
        "WIGOS Station Identifier (WSI) – a globally unique station ID used in "
        "the WMO Integrated Global Observing System. Example: '0-203-0-10702039001'."
    ),
    "GH_ID": (
        "GH station identifier – a compact local code for the same station in the GH "
        "database (e.g. 'C2LEDE01'). It is stable across time, while metadata such as "
        "location or name may change in different records."
    ),
    "BEGIN_DATE": (
        "Start of the validity period for this station metadata record "
        "(ISO 8601 UTC, e.g. '2016-02-01T00:00:00Z')."
    ),
    "END_DATE": (
        "End of the validity period for this station metadata record "
        "(ISO 8601 UTC, e.g. '2019-02-28T23:59:00Z'). "
        "The value '3999-12-31T23:59:00Z' is used to indicate an open-ended/ongoing period."
    ),
    "FULL_NAME": (
        "Full station name as used in the metadata, possibly including locality "
        "qualifiers (e.g. 'Ševětín, Mazelov')."
    ),
    "GEOGR1": (
        "Station longitude in decimal degrees East (WGS84), e.g. 14.6184119."
    ),
    "GEOGR2": (
        "Station latitude in decimal degrees North (WGS84), e.g. 48.9334328."
    ),
    "ELEVATION": (
        "Station elevation above mean sea level in metres, e.g. 496.0."
    ),
}
    
    
# station - quantity
meta2 = {
    "OBS_TYPE": (
        "Type of observation series. For example, 'DLY' = daily aggregated values "
        "derived from sub-daily measurements."
    ),
    "WSI": (
        "WIGOS Station Identifier – globally unique station ID "
        "for which this observation series is defined (e.g. '0-20000-0-11406')."
    ),
    "BEGIN_DATE": (
        "Start of the validity period for this observation/element configuration "
        "(ISO 8601 UTC, e.g. '1961-01-01T00:00:00Z')."
    ),
    "END_DATE": (
        "End of the validity period for this observation/element configuration "
        "(ISO 8601 UTC). The value '3999-12-31T23:59:00Z' denotes an open-ended period."
    ),
    "EG_EL_ABBREVIATION": (
        "Abbreviation (code) of the observed element, common across stations. "
        "Examples: 'T' = air temperature, 'TMI' = minimum temperature, 'SRA' = precipitation, "
        "'SSV' = sunshine duration, 'F' = wind speed, 'H' = relative humidity, etc."
    ),
    "NAME": (
        "Human-readable name of the observed element, usually in Czech "
        "(e.g. 'Teplota', 'Srážka', 'Sluneční svit')."
    ),
    "UN_DESCRIPTION": (
        "Unit of the observed element, given as a short string, typically SI or derived "
        "(e.g. '°C', 'mm', 'm/s', 'hPa', 'hod', '%', 'cm')."
    ),
    "HEIGHT": (
        "Height or depth of the sensor relative to the ground surface, in metres. "
        "Positive values = height above ground (e.g. 2.0 m for air temperature, 10.0 m for wind); "
        "negative values = depth below surface (e.g. −0.1 m for soil temperature at 10 cm); "
        "0.0 = at surface (e.g. snow depth, evaporation pan)."
    ),
    "SCHEDULE": (
        "Schedule of measurement times used to derive the daily value, as a comma-separated list. "
        "Examples: '06:00,13:00,20:00,AVG' = daily statistic from those hours; "
        "'20:00' = daily max/min at 20 UTC; '00:00' = daily total/extreme for the whole day."
    ),
}
    
