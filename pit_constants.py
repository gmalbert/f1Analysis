# Typical total pit lane times (entry to exit, seconds) by grandPrixId (2015–present)
# 2023 average was 24.62 (will be used when no other information is available)
PIT_LANE_TIME_S = {
    'australia': 19.3,
    'bahrain': 22.9,
    'saudi_arabia': 19.2,
    'azerbaijan': 20.0,
    'miami': 19.9,
    'emilia-romagna': 28.2,  # Imola
    'monaco': 19.4,
    'spain': 22.2,
    'canada': 18.4,
    'austria': 20.3,
    'britain': 19.9,
    'hungary': 20.0,
    'belgium': 18.8,
    'netherlands': 20.6,
    'italy': 18.0,  # Monza
    'singapore': 27.0,
    'japan': 22.7,
    'usa': 20.0,  # Austin
    'mexico': 20.0,
    'brazil': 21.0,
    'abu_dhabi': 22.0,
    'qatar': 21.0,
    'russia': 21.0,
    'turkey': 21.0,
    'portugal': 21.0,
    'france': 20.0,
    'sakhir': 21.0,
    'styria': 18.0,
    'eifel': 21.0,
    'tuscany': 21.0,
    '70th_anniversary': 21.0,
    'europe': 20.0,
    'malaysia': 21.0,
    'china': 23.9,
    'india': 21.0,
    'korea': 21.0,
    'germany': 21.0,
    'pacific': 21.0,
    'south_africa': 21.0,
    'san_marino': 25.0,
    'usa_indianapolis': 21.0,
    '70th-anniversary': 29.6,  # TODO: Fill in actual value
    'abu-dhabi': 22.0,  # TODO: Fill in actual value
    'argentina': 24.62,  # Unknown, using 2023 average
    'caesars-palace': 22.32,  # TODO: Fill in actual value
    'dallas': 24.0,  # TODO: Fill in actual value
    'detroit': 24.62,  # TODO: Unknown, using 2023 average
    'great-britain': 21.0,  # TODO: Fill in actual value
    'indianapolis': 24.0,  # TODO: Fill in actual value
    'las-vegas': 20.0,  # Updated based on 2023-2024 pit stop data
    'luxembourg': 24.62,  # TODO: Unknown, using 2023 average
    'morocco': 24.62,  # TODO: Unknown, using 2023 average
    'pescara': 24.62,  # TODO: Unknown, using 2023 average
    'san-marino': 27.0,  # TODO: Fill in actual value
    'sao-paulo': 23.73,  # TODO: Fill in actual value
    'saudi-arabia': 22.17,  # TODO: Fill in actual value
    'south-africa': 21.0,  # TODO: Fill in actual value
    'sweden': 24.62,  # TODO: Unknown, using 2023 average
    'switzerland': 24.62,  # TODO: Unknown, using 2023 average
    'united-states': 23.54,  # TODO: Fill in actual value
    'united-states-west': 20.0,  # Updated based on 2023-2024 pit stop data - Las Vegas
    # Add/adjust as needed for new/old circuits
}

# Typical stationary time for tyre change (seconds)
TYPICAL_STATIONARY_TIME_S = 2.5